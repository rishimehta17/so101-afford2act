"""Step 7 (paper: data_generation/train_bc_ang.py): train the keypoint policy from scratch.

    python 7_train.py --data data/mug_a2a --name mug40fk_gl_h            # the model that lifted the mug (~16 min)
    python 7_train.py --data data/mug_a2a --name test --val ep_015 ep_018 # keep demos out for validation

The paper's model and training: tokens -> Linear(->256) -> TransformerEncoder (3 layers, 4 heads, d_ff 512,
dropout 0.1) -> learned gate (softmax weight per token) -> weighted sum -> shared MLP -> two heads: the next 10 actions
(standardised) and the next 10 gripper-closed probabilities. Loss MSE + BCE; Adam, lr 5e-5, batch 128, 1000 epochs,
gradient clip 1.0. No token-id embedding (the paper's is defined but unused; kept so checkpoints load).
What this repo uses (see NOTES_vs_paper.md):
  - tokens (x px, y px), as in the paper's text: 4 gripper points (from the joints) + 15 handle points; no frame number
  - --history 0 5 10: the 19 points of the current frame and of 5 and 10 frames ago (0.5 / 1.0 s at 10 Hz) = 57 tokens,
    each tagged with a learned "which frame" embedding (the authors' KP_WINDOW idea, with spaced frames); before the
    demo start the first frame is repeated
  - --segment grasp_lift: samples from the start up to the top of the lift (the paper's task "grasp-and-lift"); top =
    highest gripper point (ee_z) between the grasp (first open->closed of gz) and the release
  - action = per-step change of the 5 measured follower arm joints (deg), not the gripper-pose change of the paper
  - training windows stay inside one demo; validation = whole held-out demo(s); scalers fitted on the training demos
    only; fixed seed; nn.TransformerEncoder(enable_nested_tensor=False) (otherwise PyTorch's eval fast path drops padded
    tokens and the gate's shapes no longer match)
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import ARM, HERE

K_POINTS, ACTION_WINDOW = 19, 10
EE_COLS = [f"ee_{a}{i}" for i in range(1, 5) for a in "xy"]
OBJ_COLS = [f"obj_{a}{i}" for i in range(1, 16) for a in "xy"]

class TinyTransformer(nn.Module):
    """The paper's model (train_bc_ang.py), KP_WINDOW = 1; n_hist > 1 = several frames (19 tokens each)."""

    def __init__(self, k_points=K_POINTS, action_window=ACTION_WINDOW, d_model=256, nhead=4, layers=3, d_ff=512,
                 action_dim=6, token_dim=3, n_hist=1):
        super().__init__()
        self.K, self.n_hist = k_points, n_hist
        if n_hist > 1:  # which past frame a token comes from (tokens are ordered frame by frame, 19 each)
            self.hist_emb = nn.Embedding(n_hist, d_model)
            self.register_buffer("hist_ids", torch.arange(n_hist).repeat_interleave(k_points), persistent=False)
        self.embed = nn.Linear(token_dim, d_model)  # 3 = (x, y, frame number) as the code; 2 = (x, y) as the text
        self.kp_emb = nn.Embedding(k_points, d_model)  # defined but NOT used, exactly as in the paper
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=0.1,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.gate = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Linear(2 * d_model, 1))
        self.shared = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(d_ff, d_ff), nn.GELU(), nn.Dropout(0.1))
        self.head_reg = nn.Linear(d_ff, action_dim * action_window)  # 5 arm-joint changes per step
        self.head_grip = nn.Sequential(nn.Linear(d_ff, action_window), nn.Sigmoid())

    def forward(self, x, mask, return_weights=False):
        """x [B,K,3] standardised tokens; mask [B,K] True = no point."""
        e = self.embed(x)
        if self.n_hist > 1:
            e = e + self.hist_emb(self.hist_ids)[None]
        z = self.encoder(e, src_key_padding_mask=mask)
        logits = self.gate(z).squeeze(-1).masked_fill(mask, -1e9)
        w = torch.softmax(logits, dim=1)
        h = self.shared(torch.bmm(w.unsqueeze(1), z).squeeze(1)) #Make this as an input for ACT
        out = self.head_reg(h), self.head_grip(h)
        return (*out, w) if return_weights else out



def lift_top(gz, z):
    """index of the top of the lift: highest ee_z between the grasp (first open->closed) and the release."""
    cl = np.where((gz[:-1] == 0) & (gz[1:] == 1))[0] + 1
    op = np.where((gz[:-1] == 1) & (gz[1:] == 0))[0] + 1
    g = cl[0]
    rel = op[op > g][0] if (op > g).any() else len(gz)
    return int(g + np.argmax(z[g:rel]))


def load_demo(ep, segment="grasp_lift"):
    """-> tokens (T,19,2) raw px with NaN for missing, joint changes (T-1,5) of the follower's MEASURED arm joints
    (deg), gripper-closed (T,), number of samples to use (segment)."""
    m = pd.read_csv(ep / "merged_coords.csv")
    xy = m[EE_COLS + OBJ_COLS].to_numpy(float).reshape(len(m), K_POINTS, 2)
    xy[xy == -1] = np.nan                                   # tracks.csv uses -1 for empty point slots
    tok = xy
    states = list(csv.DictReader(open(ep / "robot_states.csv")))
    assert [int(s["frame"]) for s in states] == list(m["frame"]), f"{ep.name}: frames differ"
    gz = np.array([int(s["gz"]) for s in states], float)
    last = len(gz) if segment == "full" else lift_top(gz, np.array([float(s["ee_z"]) for s in states])) + 1
    q = np.array([[float(s[f"follower_{j}"]) for j in ARM] for s in states])
    return tok, np.diff(q, axis=0), gz, last


def with_history(tok, i, lags=(0,)):
    """tokens of frames i - lag for each lag, stacked frame by frame -> (19 * len(lags), dim); before the start the
    first frame is repeated"""
    return np.concatenate([tok[max(i - L, 0)] for L in lags], axis=0)


def windows(tok, d, gz, last=None, aw=ACTION_WINDOW, lags=(0,)):
    """Paper's pairing inside one demo: tokens of frame i -> joint changes i..i+aw-1 (change i = frame i -> i+1)
    and gripper-closed of frames i+1..i+aw. Only samples i < last (grasp_lift: up to the top of the lift; their
    labels may run up to 9 frames past it). lags: also the tokens of frames i - lag (history)."""
    n = len(d) - aw + 1
    n = n if last is None else min(n, last)
    return (np.stack([with_history(tok, i, lags) for i in range(n)]), np.stack([d[i:i + aw] for i in range(n)]),
            np.stack([gz[i + 1:i + 1 + aw] for i in range(n)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/mug_a2a"))
    p.add_argument("--name", required=True, help="models/<name>/")
    p.add_argument("--val", nargs="+", default=["none"], help="held-out demo(s); none = train on all (default)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="1", help="CUDA device (default 1: GPU 0 of the lab PC is shared)")
    p.add_argument("--segment", choices=["full", "grasp_lift"], default="grasp_lift")
    p.add_argument("--history", type=int, nargs="+", default=[0, 5, 10],
                   help="frames back whose 19 points are also given (0 = the current frame)")
    a = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps = sorted(e for e in Path(a.data).glob("ep_*") if (e / "merged_coords.csv").exists())
    val_names = [] if a.val == ["none"] else a.val
    sets = {"train": [e for e in eps if e.name not in val_names], "val": [e for e in eps if e.name in val_names]}
    tdim = 2
    data = {}
    for split, demos in sets.items():
        assert a.history[0] == 0, "--history must start with 0 (the current frame)"
        parts = [windows(*load_demo(e, a.segment), lags=tuple(a.history)) for e in demos]
        data[split] = [np.concatenate(x) for x in zip(*parts)] if parts else None
        print(f"{split}: {[e.name for e in demos]} -> {0 if not parts else len(data[split][0])} samples")

    # scalers on the training demos only (tokens: valid tokens only, as the paper)
    Xtr = data["train"][0].reshape(-1, tdim)
    ok = ~np.isnan(Xtr).any(1)
    x_mean, x_std = Xtr[ok].mean(0), Xtr[ok].std(0)
    adim = 5
    a_flat = data["train"][1].reshape(-1, adim)
    a_mean, a_std = a_flat.mean(0), a_flat.std(0)

    def tensors(split):
        X, D, G = data[split]
        mask = np.isnan(X).any(-1)
        Xn = np.nan_to_num((X - x_mean) / x_std, nan=0.0)
        Dn = ((D - a_mean) / a_std).reshape(len(D), -1)
        return [torch.tensor(v, dtype=t) for v, t in ((Xn, torch.float32), (mask, torch.bool),
                                                      (Dn, torch.float32), (G, torch.float32))]

    tr = tensors("train")
    va = tensors("val") if data["val"] is not None else None
    model = TinyTransformer(action_dim=adim, token_dim=tdim, n_hist=len(a.history)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    mse, bce = nn.MSELoss(), nn.BCELoss()

    def run(ts, train):
        model.train(train)
        idx = torch.randperm(len(ts[0])) if train else torch.arange(len(ts[0]))
        tot = {"loss": 0.0, "mse": 0.0, "bce": 0.0}
        for b in range(0, len(idx), a.batch):
            X, M, D, G = (t[idx[b:b + a.batch]].to(dev) for t in ts)
            with torch.set_grad_enabled(train):
                pr, pg = model(X, M)
                lm, lb = mse(pr, D), bce(pg, G)
                loss = lm + lb
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            for k, v in (("loss", loss), ("mse", lm), ("bce", lb)):
                tot[k] += v.item() * len(X)
        return {k: v / len(idx) for k, v in tot.items()}

    out = Path(a.name) if "/" in a.name else HERE / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"K": K_POINTS, "ACTION_WINDOW": ACTION_WINDOW, "D_MODEL": 256, "NHEAD": 4, "LAYERS": 3, "D_FF": 512,
           "ACTION": "joints", "ACTION_DIM": adim, "TOKEN_DIM": tdim, "TIME": "none", "SEGMENT": a.segment,
           "HISTORY": a.history,
           "action": "per-step change of the 5 measured follower joints (deg): " + " ".join(ARM)
                     + "; gripper = P(closed)",
           "train_demos": [e.name for e in sets["train"]], "val_demos": [e.name for e in sets["val"]],
           "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed, "data": Path(a.data).name}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    def save(path):
        torch.save({"model_state": model.state_dict(), "scaler_state": (x_mean, x_std),
                    "act_scaler_state": (a_mean, a_std), "config": cfg}, path)

    best, log, t0 = float("inf"), [], time.time()
    for epoch in range(1, a.epochs + 1):
        r_tr = run(tr, True)
        r_va = run(va, False) if va else {"loss": float("nan"), "mse": float("nan"), "bce": float("nan")}
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in r_tr.items()}, **{f"val_{k}": v for k, v in r_va.items()}})
        if va and r_va["loss"] < best:
            best = r_va["loss"]
            save(out / "best_bc_model.pt")
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch {epoch:4d} | train {r_tr['loss']:.4f} (mse {r_tr['mse']:.4f} bce {r_tr['bce']:.4f}) | "
                  f"val {r_va['loss']:.4f} (mse {r_va['mse']:.4f} bce {r_va['bce']:.4f}) | {time.time() - t0:.0f} s",
                  flush=True)
    save(out / "final_bc_model.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, k in enumerate(("loss", "mse", "bce")):
        ax[i].plot(df["epoch"], df[f"train_{k}"], label="train")
        if va:
            ax[i].plot(df["epoch"], df[f"val_{k}"], label="val " + ",".join(cfg["val_demos"]))
        ax[i].set_title(k), ax[i].set_xlabel("epoch"), ax[i].set_yscale("log"), ax[i].legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=80)
    print(f"saved -> {out} (best val {best:.4f})")


if __name__ == "__main__":
    main()
