"""
Q2 - hardware-aware evolutionary Neural Architecture Search on CIFAR-10.

Algorithm ((mu + lambda) evolution with tournament selection):
  1. Initial population: sample random architectures. Each is built, measured on the CPU (latency, size,
     peak memory) and REJECTED immediately if it breaks a deployment constraint; feasible ones are trained
     for a few epochs on a CIFAR-10 training subset and scored on a held-out validation split (the official
     test set is never used during the search).
  2. Each generation: pick parents by tournament selection, create children by uniform crossover + mutation,
     measure -> reject / train -> score them, then keep the best `population` individuals (elitism).
  3. Fitness (MnasNet-style soft latency reward on top of the hard constraints):
         fitness = val_acc * (latency / latency_target) ** w,   w = -0.07
     so among equally accurate networks the faster one wins.
Every candidate (accepted or rejected) is appended to results/nas_history.csv / .json as it is evaluated.
The best architecture is written to models/nas_best_config.json and results/nas_best_architecture.json.

    python src/nas_search.py                                     # defaults below
    python src/nas_search.py --population 12 --generations 10 --max-latency-ms 2.5
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from common import (NAS_BEST_CONFIG, rel, RESULTS_DIR, banner, default_workers, fit, get_device, get_loaders,
                    seed_everything, write_csv, write_json)
from nas_generator import build_network
from nas_hardware import Constraints, measure_candidate
from nas_search_space import ArchConfig, crossover, mutate, sample_config


class EvolutionarySearch:
    def __init__(self, args, constraints: Constraints, device):
        self.args, self.c, self.device = args, constraints, device
        self.rng = random.Random(args.seed)
        self.history: list[dict] = []
        self.cache: dict[str, dict] = {}       # config key -> evaluated record (avoid retraining duplicates)
        self.loaders: dict[int, tuple] = {}    # resolution -> (train, val) loaders
        self.t_start = time.time()

    # ------------------------------------------------------------------ helpers
    def loaders_for(self, res: int):
        if res not in self.loaders:
            a = self.args
            self.loaders[res] = get_loaders("cifar10", a.batch_size, a.workers, resolution=res,
                                            train_subset=a.train_subset, val_split=a.val_split,
                                            synthetic=a.synthetic, seed=a.seed)
        return self.loaders[res]

    def fitness(self, acc: float, lat: float) -> float:
        return acc * (lat / self.args.latency_target_ms) ** self.args.reward_w

    def save_history(self):
        write_csv(RESULTS_DIR / "nas_history.csv", [{k: v for k, v in h.items() if k != "config"} | {
            "config": h["config_summary"]} for h in self.history])
        write_json(RESULTS_DIR / "nas_history.json", self.history)

    # ------------------------------------------------------------------ evaluation of one candidate
    def evaluate(self, cfg: ArchConfig, generation: int, origin: str) -> dict:
        key = cfg.key()
        if key in self.cache:
            return self.cache[key]
        idx = len(self.history)
        net = build_network(cfg, 10)
        prof = measure_candidate(net, cfg.resolution, self.c)
        ok, reasons = self.c.check(prof)
        rec = {"id": idx, "generation": generation, "origin": origin, "status": "accepted" if ok else "rejected",
               "reject_reasons": "; ".join(reasons), "resolution": cfg.resolution,
               "params": prof["params"], "macs": prof["macs"], "size_mb": round(prof["size_mb"], 4),
               "latency_ms": round(prof["latency_ms"], 4),
               "peak_mem_mb": None if prof["peak_mem_mb"] is None else round(prof["peak_mem_mb"], 3),
               "val_acc": None, "fitness": None, "train_time_s": None,
               "config_summary": cfg.summary(), "config": cfg.to_dict()}
        tag = f"cand {idx:3d} g{generation} {origin:9s}"
        if not ok:
            print(f"[{tag}] REJECTED ({rec['reject_reasons']}) | {cfg.summary()}", flush=True)
        else:
            print(f"[{tag}] measured: {prof['latency_ms']:.2f} ms, {prof['size_mb']:.2f} MB, "
                  f"{prof['params'] / 1e3:.0f} K params -> training | {cfg.summary()}", flush=True)
            train_loader, val_loader = self.loaders_for(cfg.resolution)
            torch.manual_seed(self.args.seed)
            t0 = time.time()
            res = fit(net, train_loader, val_loader, self.device, self.args.epochs_per_candidate, self.args.lr,
                      5e-4, tag=f"cand {idx}", verbose=self.args.verbose)
            rec["val_acc"] = round(res["best_acc"], 3)
            rec["fitness"] = round(self.fitness(res["best_acc"], prof["latency_ms"]), 4)
            rec["train_time_s"] = round(time.time() - t0, 1)
            print(f"[{tag}] val acc {rec['val_acc']:.2f}% | fitness {rec['fitness']:.3f} | "
                  f"{rec['train_time_s']:.0f}s", flush=True)
        self.history.append(rec)
        self.cache[key] = rec
        self.save_history()
        return rec

    def feasible_candidate(self, make, generation: int, origin: str) -> dict | None:
        """Generate configs with `make()` until one passes the constraints (or attempts run out)."""
        for _ in range(self.args.max_attempts):
            cfg = make()
            if cfg.key() in self.cache:
                continue
            rec = self.evaluate(cfg, generation, origin)
            if rec["status"] == "accepted":
                return rec
        return None

    def tournament(self, pop: list[dict]) -> ArchConfig:
        group = self.rng.sample(pop, min(self.args.tournament, len(pop)))
        return ArchConfig.from_dict(max(group, key=lambda r: r["fitness"])["config"])

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        a = self.args
        banner(f"Generation 0: random initial population of {a.population}")
        pop = []
        while len(pop) < a.population:
            rec = self.feasible_candidate(lambda: sample_config(self.rng), 0, "random")
            if rec is None:
                raise RuntimeError("Could not find feasible architectures - relax the constraints.")
            pop.append(rec)

        for g in range(1, a.generations + 1):
            banner(f"Generation {g}/{a.generations}")
            children = []
            for _ in range(a.children):
                def make():
                    p1, p2 = self.tournament(pop), self.tournament(pop)
                    child = crossover(p1, p2, self.rng) if self.rng.random() < a.crossover_prob else p1
                    return mutate(child, self.rng, a.mutation_prob)
                rec = self.feasible_candidate(make, g, "offspring")
                if rec is not None:
                    children.append(rec)
            pop = sorted({r["id"]: r for r in pop + children}.values(), key=lambda r: r["fitness"],
                         reverse=True)[:a.population]
            best = pop[0]
            print(f"--> generation {g} best: id {best['id']} | val {best['val_acc']:.2f}% | "
                  f"{best['latency_ms']:.2f} ms | {best['size_mb']:.2f} MB | fitness {best['fitness']:.3f} | "
                  f"elapsed {(time.time() - self.t_start) / 60:.1f} min", flush=True)
        return pop[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--children", type=int, default=6, help="offspring per generation (lambda)")
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--crossover-prob", type=float, default=0.5)
    p.add_argument("--mutation-prob", type=float, default=0.25)
    p.add_argument("--max-attempts", type=int, default=30, help="tries to produce one feasible candidate")
    p.add_argument("--epochs-per-candidate", type=int, default=5)
    p.add_argument("--train-subset", type=int, default=20000, help="training images per candidate (proxy task)")
    p.add_argument("--val-split", type=int, default=5000, help="held-out validation images from the train set")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="random data with CIFAR shapes (smoke test)")
    p.add_argument("--verbose", action="store_true", help="print every training epoch of every candidate")
    # deployment constraints and reward
    p.add_argument("--max-latency-ms", type=float, default=3.0)
    p.add_argument("--max-size-mb", type=float, default=1.5)
    p.add_argument("--max-peak-mem-mb", type=float, default=2.5)
    p.add_argument("--latency-threads", type=int, default=1)
    p.add_argument("--latency-target-ms", type=float, default=1.5, help="T in the soft latency reward")
    p.add_argument("--reward-w", type=float, default=-0.07, help="w in the soft latency reward")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    c = Constraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb, args.latency_threads)
    banner(f"Hardware-aware evolutionary NAS on CIFAR-10 | device={device} | constraints: {c.to_dict()}"
           + (" | SYNTHETIC DATA (smoke test)" if args.synthetic else ""))
    search = EvolutionarySearch(args, c, device)
    best = search.run()

    n_acc = sum(h["status"] == "accepted" for h in search.history)
    n_rej = len(search.history) - n_acc
    best_out = {"id": best["id"], "val_acc": best["val_acc"], "fitness": best["fitness"],
                "latency_ms": best["latency_ms"], "size_mb": best["size_mb"], "peak_mem_mb": best["peak_mem_mb"],
                "params": best["params"], "macs": best["macs"], "summary": best["config_summary"],
                "config": best["config"], "constraints": c.to_dict(), "search_args": vars(args),
                "evaluated": len(search.history), "accepted": n_acc, "rejected": n_rej,
                "search_time_min": round((time.time() - search.t_start) / 60, 2)}
    write_json(NAS_BEST_CONFIG, best["config"])
    write_json(RESULTS_DIR / "nas_best_architecture.json", best_out)
    banner("Search finished")
    print(f"candidates evaluated: {len(search.history)} (accepted {n_acc}, rejected by constraints {n_rej})")
    print(f"best: id {best['id']} | val acc {best['val_acc']:.2f}% | {best['latency_ms']:.2f} ms | "
          f"{best['size_mb']:.3f} MB | peak mem {best['peak_mem_mb']} MB\n      {best['config_summary']}")
    print(f"saved {rel(NAS_BEST_CONFIG)} and results/nas_best_architecture.json, history in results/nas_history.csv/.json")


if __name__ == "__main__":
    main()
