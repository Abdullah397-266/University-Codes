"""
Deliverable 5: structured CHANNEL pruning of ResNet-18 with real architecture reconstruction.

Which channels are pruned?  For every BasicBlock we rank the filters of conv1 by their L1 norm
(Li et al., "Pruning Filters for Efficient ConvNets") and keep the strongest ones. Removing filter k of
conv1 forces three coupled edits so tensor shapes stay valid:
    conv1.weight[k]   (output channel)      -> removed
    bn1.{weight,bias,running_mean,running_var}[k]   -> removed
    conv2.weight[:, k] (input channel)      -> removed
The block's OUTPUT width is untouched, so the residual addition (which needs identical shapes on both
branches) stays valid and nothing has to be changed in the next block.
A new, physically smaller ResNet18(mid_channels=[...]) is built and the surviving weights are copied in,
so no masked zero-channels remain. The new model is then fine-tuned.
"""
import argparse

import torch

from common import (MODEL_FILES, ResNet18, count_macs, count_params, evaluate, get_datasets, get_device,
                    load_float_model, make_loader, save_checkpoint, save_history, seed_everything,
                    train_epochs)


def build_pruned_model(old: ResNet18, ratio: float, min_channels: int = 8) -> ResNet18:
    old_sd = old.state_dict()
    keeps, mids = [], []
    for i, blk in enumerate(old.layers):
        importance = blk.conv1.weight.detach().abs().sum(dim=(1, 2, 3))           # L1 norm per filter
        k = max(min_channels, int(round(importance.numel() * (1 - ratio))))
        keep = torch.topk(importance, k).indices.sort().values                   # keep top-k, original order
        keeps.append(keep)
        mids.append(k)

    new = ResNet18(mid_channels=mids)
    new_sd = {k: v.clone() for k, v in old_sd.items()}
    for i, keep in enumerate(keeps):
        p = f"layers.{i}."
        new_sd[p + "conv1.weight"] = old_sd[p + "conv1.weight"][keep].clone()
        for s in ("weight", "bias", "running_mean", "running_var"):
            new_sd[p + f"bn1.{s}"] = old_sd[p + f"bn1.{s}"][keep].clone()
        new_sd[p + "conv2.weight"] = old_sd[p + "conv2.weight"][:, keep].clone()
    new.load_state_dict(new_sd)          # raises if any shape is inconsistent
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune-ratio", type=float, default=0.5, help="fraction of inner block channels removed")
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    old, _ = load_float_model(MODEL_FILES["fp32"], "cpu")
    train, _, test = get_datasets()
    train_loader = make_loader(train, args.batch_size, shuffle=True, workers=args.workers)
    test_loader = make_loader(test, 256, workers=args.workers)

    base_acc = evaluate(old.to(device), test_loader, device)["acc"]
    old.cpu()
    new = build_pruned_model(old, args.prune_ratio)
    print(f"Inner channels per block: {old.mid_channels} -> {new.mid_channels}")
    p0, p1 = count_params(old)[0], count_params(new)[0]
    m0, m1 = count_macs(old), count_macs(new)
    print(f"Params {p0:,} -> {p1:,} ({100 * (1 - p1 / p0):.1f}% fewer) | MACs {m0 / 1e6:.0f}M -> {m1 / 1e6:.0f}M")

    new.to(device)
    acc_before = evaluate(new, test_loader, device)["acc"]
    print(f"Accuracy: baseline {base_acc:.2f}% | right after channel pruning {acc_before:.2f}%")

    history = train_epochs(new, train_loader, test_loader, device, args.ft_epochs, args.lr,
                           tag="structured", amp=True)
    acc = evaluate(new, test_loader, device)["acc"]
    save_checkpoint(MODEL_FILES["structured_pruned"], new.cpu(), test_acc=acc, acc_before_finetune=acc_before,
                    prune_ratio=args.prune_ratio)
    save_history(history, "structured")
    print(f"Fine-tuned accuracy: {acc:.2f}%  -> saved {MODEL_FILES['structured_pruned']}")


if __name__ == "__main__":
    main()
