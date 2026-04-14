#!/usr/bin/env python3
"""Phase 0b: Pre-align BioMistral Adapter MLP using STEGO self-distillation.

Trains the BioMistralAdapterMLP (4096→768) to preserve BioMistral's inter-class
semantic structure using STEGO correspondence loss (Hamilton et al., ICLR 2022).

All 31 classes (20 base + 11 novel) participate in pre-alignment to maintain
complete inter-class topology for zero-shot novel detection.

Usage:
    python scripts/prealign_adapter.py
    python scripts/prealign_adapter.py --embeddings embeddings/biomistral_tct_ngc.pt \
                                       --output embeddings/adapter_prealigned.pth \
                                       --steps 1000 --negative-pressure 0.4

Output:
    adapter_prealigned.pth — state_dict for BioMistralAdapterMLP
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


def stego_corr_loss(student_feats, teacher_sim, b=0.4):
    """STEGO correspondence distillation loss.

    L = -mean(clamp(S_ij, 0, 0.8) * (F_ij - b))

    Args:
        student_feats: [C, D] L2-normalized student features.
        teacher_sim: [C, C] teacher pairwise cosine similarity.
        b: negative pressure (shift).

    Returns:
        Scalar loss.
    """
    S = torch.mm(student_feats, student_feats.t())
    C = S.shape[0]
    mask = ~torch.eye(C, dtype=torch.bool, device=S.device)
    F_off = teacher_sim[mask]
    S_off = S[mask]
    loss = -(torch.clamp(S_off, min=0.0, max=0.8) * (F_off - b)).mean()
    return loss


def variance_reg(feats, gamma=0.1):
    """VICReg-inspired variance regularization to prevent collapse.

    Args:
        feats: [C, D] features.
        gamma: target std threshold.

    Returns:
        Scalar regularization loss.
    """
    std = feats.std(dim=0)
    return torch.clamp(gamma - std, min=0).mean()


def main():
    parser = argparse.ArgumentParser(description="Phase 0b: STEGO pre-alignment for Adapter MLP")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="embeddings/biomistral_tct_ngc.pt",
        help="Path to BioMistral embeddings .pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings/adapter_prealigned.pth",
        help="Output path for pre-aligned adapter weights",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--negative-pressure", type=float, default=0.4)
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # Adapter architecture (must match BioMistralAdapterMLP)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--output-dim", type=int, default=768)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load BioMistral embeddings
    data = torch.load(args.embeddings, map_location="cpu")
    bio_t = data["embeddings"].float()  # [C, 4096]
    categories = data["categories"]
    C = bio_t.shape[0]
    print(f"Loaded {C} class embeddings: {bio_t.shape}")
    print(f"  Base: {data.get('num_base', '?')}, Novel: {data.get('num_novel', '?')}")

    # Precompute teacher similarity
    bio_norm = F.normalize(bio_t, dim=-1)
    teacher_sim = torch.mm(bio_norm, bio_norm.t())  # [C, C]

    mask = ~torch.eye(C, dtype=torch.bool)
    sim_off = teacher_sim[mask]
    print(f"\nTeacher similarity: mean={sim_off.mean():.4f}, "
          f"min={sim_off.min():.4f}, max={sim_off.max():.4f}")

    # Build Adapter MLP (same architecture as BioMistralAdapterMLP)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from detrex.layers.biomistral_adapter import BioMistralAdapterMLP

    adapter = BioMistralAdapterMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=0.0,  # No dropout during pre-alignment
    )

    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-5
    )

    print(f"\nPre-aligning Adapter MLP: {args.steps} steps, lr={args.lr}")
    print(f"  Architecture: {args.input_dim} → {args.hidden_dim} → {args.output_dim}")
    print(f"  Negative pressure b={args.negative_pressure}")
    print("-" * 60)

    best_rho = -1
    best_state = None

    for step in range(1, args.steps + 1):
        adapter.train()
        optimizer.zero_grad()

        projected = adapter(bio_t)  # [C, 768]
        proj_norm = F.normalize(projected, p=2, dim=-1)

        loss_corr = stego_corr_loss(proj_norm, teacher_sim, b=args.negative_pressure)
        loss_var = variance_reg(projected, gamma=0.1)
        loss = loss_corr + args.var_weight * loss_var

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == 1:
            adapter.eval()
            with torch.no_grad():
                proj = adapter(bio_t)
                proj_n = F.normalize(proj, p=2, dim=-1)
                S = torch.mm(proj_n, proj_n.t())
                student_off = S[mask].numpy()
                teacher_off = teacher_sim[mask].numpy()
                rho, _ = spearmanr(student_off, teacher_off)
                s_off = S[mask]

            print(
                f"Step {step:5d} | "
                f"Loss={loss.item():.6f} (corr={loss_corr.item():.4f}, var={loss_var.item():.4f}) | "
                f"Spearman={rho:.4f} | "
                f"S: mean={s_off.mean():.4f} min={s_off.min():.4f} max={s_off.max():.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if rho > best_rho:
                best_rho = rho
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

    # Save best checkpoint
    print(f"\n{'=' * 60}")
    print(f"Best Spearman correlation: {best_rho:.4f}")

    # Final evaluation
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        proj = adapter(bio_t)
        proj_n = F.normalize(proj, p=2, dim=-1)
        S = torch.mm(proj_n, proj_n.t())
        student_off = S[mask].numpy()
        teacher_off = teacher_sim[mask].numpy()
        rho, _ = spearmanr(student_off, teacher_off)

    print(f"\nFinal metrics (best checkpoint):")
    print(f"  Spearman (structure preservation): {rho:.4f}")
    print(f"  Student similarity: mean={S[mask].mean():.4f}, std={S[mask].std():.4f}")

    torch.save(best_state, args.output)
    print(f"\nSaved pre-aligned adapter to {args.output}")


if __name__ == "__main__":
    main()
