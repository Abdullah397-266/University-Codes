"""
Deliverable 4: iterative, global, magnitude-based UNSTRUCTURED pruning of ResNet-18.

* Prunes every Conv2d and Linear weight (global L1-magnitude ranking).
* Iterative: sparsity follows a cubic schedule over N iterations; each iteration is followed by fine-tuning
  (masks stay active, so pruned weights remain zero).
* At the end the masks are made permanent with prune.remove() (weight_orig * mask -> weight).
Unstructured sparsity shrinks the *compressed* size and non-zero parameter count, but dense CPU/GPU kernels
do not get faster, because the tensors keep their original shape.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (MODEL_FILES, count_macs, count_params, evaluate, get_datasets, get_device, load_float_model,
                    make_loader, save_checkpoint, save_history, seed_everything, train_epochs)


def prunable_layers(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_report(model):
    total = zeros = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            z, n = int((m.weight == 0).sum()), m.weight.numel()
            total, zeros = total + n, zeros + z
            print(f"  {name or 'root':28s} {100 * z / n:6.2f}% sparse  ({n} weights)")
    print(f"  GLOBAL sparsity of conv/linear weights: {100 * zeros / total:.2f}%")
    return 100 * zeros / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-sparsity", type=float, default=0.80)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=3, help="fine-tune epochs after EACH pruning iteration")
    ap.add_argument("--final-ft-epochs", type=int, default=5, help="extra fine-tune epochs after the last iteration")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    model, ck = load_float_model(MODEL_FILES["fp32"], device)
    train, _, test = get_datasets()
    train_loader = make_loader(train, args.batch_size, shuffle=True, workers=args.workers)
    test_loader = make_loader(test, 256, workers=args.workers)
    base_acc = evaluate(model, test_loader, device)["acc"]
    print(f"Baseline FP32 accuracy: {base_acc:.2f}%")

    layers = prunable_layers(model)
    history, prev = [], 0.0
    for it in range(1, args.iterations + 1):
        # cubic sparsity schedule (Zhu & Gupta, 2017): fast early, gentle at the end
        target_i = args.target_sparsity * (1 - (1 - it / args.iterations) ** 3)
        # prune.global_unstructured's `amount` is a fraction of the *remaining* weights
        amount = 1 - (1 - target_i) / (1 - prev)
        prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=amount)
        prev = target_i
        acc_pruned = evaluate(model, test_loader, device)["acc"]
        print(f"\n=== iteration {it}/{args.iterations}: target sparsity {100 * target_i:.1f}%  "
              f"acc right after pruning {acc_pruned:.2f}%")
        ft_epochs = args.ft_epochs + (args.final_ft_epochs if it == args.iterations else 0)
        h = train_epochs(model, train_loader, test_loader, device, ft_epochs, args.lr,
                         tag=f"unstruct-it{it}", amp=True)
        for row in h:
            row.update(iteration=it, target_sparsity=target_i, acc_after_pruning=acc_pruned)
        history += h

    # make pruning permanent: remove reparametrisation (weight_orig, weight_mask) -> plain zeroed weights
    for m, name in layers:
        prune.remove(m, name)
    print("\nMasks made permanent. Final layer-wise sparsity:")
    sp = sparsity_report(model)

    res = evaluate(model, test_loader, device)
    model.cpu()
    total, nonzero = count_params(model)
    save_checkpoint(MODEL_FILES["unstructured_pruned"], model, test_acc=res["acc"], sparsity=sp,
                    target_sparsity=args.target_sparsity, iterations=args.iterations)
    save_history(history, "unstructured")
    print(f"Final accuracy {res['acc']:.2f}% | non-zero params {nonzero:,}/{total:,} | "
          f"non-zero MACs {count_macs(model, nonzero_only=True) / 1e6:.1f}M")
    print(f"Saved -> {MODEL_FILES['unstructured_pruned']}")


if __name__ == "__main__":
    main()
