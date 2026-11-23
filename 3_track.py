"""Step 3 (paper: data_generation/data_generation_batch.py): 15 handle points per demo, tracked through all frames.

    python 3_track.py --data data/mug_a2a --ref ep_009        # all demos (our model used reference ep_009)
    python 3_track.py --data data/mug_a2a --ref ep_009 --eps ep_000

Per demo, on its first frame:
  1. OOAL heatmap for the affordance word(s) (default "hold") inside a fixed box (--crop), min-max normalised,
     > mask_thresh, blur, dilate -> mask
  2. box around the mask + pad -> ROI; the same for the reference demo's first frame
  3. DINO ViT-S/8 best-buddy matching between the reference ROI and this ROI (100 pairs, k-means spread)
     -> only the blob with the highest mean heat is kept
  4. keep pairs that land inside both masks; the first 15 are this demo's object points
then CoTracker3 (online) follows the 15 points through every frame.
Writes into data/<task>/<ep>/: tracks.csv (frame = image number N, obj_x1..obj_y15, -1 = no point),
visibility.csv, aff_mask_overlay.png, points_first_frame.png, track_check.png, track_params.json.
The frames are the ones listed in robot_states.csv (step 2); the images are read from the raw recording.

Uses the paper's own code, imported read-only from the authors' repo (A2A_REPO, see common.py): OOAL
(models.ooal.Net, SEEN_AFF) and d3_utils.correspondences.find_correspondences. Fixes to the paper's code (method
unchanged), see NOTES_vs_paper.md:
  - OOAL input: squash to 224x224 (OOAL's own test transform). Paper: resize 256 + centre-crop 224, then stretches
    the heatmap over the full frame -> shifted near the edges.
  - DINO gets true RGB (the paper's call path swaps red and blue: it converts to RGB, the extractor flips again).
  - CoTracker gets RGB frames with values 0-255 (the model divides by 255 itself; the paper's wrapper divides first
    -> an almost black video), called the way CoTracker's README does it (8-frame steps on a 16-frame window).
  - OOAL runs on a fixed box around the mug area (--crop), not the full frame: on the full 1280x720 frame one OOAL
    patch is ~80x45 px and the mug handle (~40 px) is smaller than a patch; the heat landed on the table.
  - tracks.csv frame numbers = image numbers (from 1), same as robot_states.csv (paper: from 0 -> off-by-one merge).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import A2A_REPO as PAPER, HERE, OOAL_CKPT  # the authors' repo (read-only) + OOAL checkpoint

sys.dont_write_bytecode = True                         # don't write __pycache__ into the authors' repo
sys.path[:0] = [str(PAPER / "OOAL"), str(PAPER)]


def load_ooal(device):
    from models.ooal import Net
    from data.agd20k_ego import SEEN_AFF
    model = Net(type("A", (), {"class_names": SEEN_AFF})(), 768, 512).to(device).eval()  # as the paper
    ckpt = torch.load(OOAL_CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model, SEEN_AFF, (len(missing), len(unexpected))


def affordance_mask(img_bgr, model, aff_names, affs, mask_thresh, dilate, crop=None):
    """Paper's compute_affordance_mask, with OOAL's own squash-to-224 input. crop = (x0, y0, x1, y1): OOAL sees
    only that box (the heatmap is put back in place; 0 outside), so a small handle covers more OOAL patches."""
    H_full, W_full = img_bgr.shape[:2]
    x0c, y0c, x1c, y1c = crop if crop else (0, 0, W_full, H_full)
    img_bgr = img_bgr[y0c:y1c, x0c:x1c]
    H, W = img_bgr.shape[:2]
    rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    dev = next(model.parameters()).device
    x = ((x - mean) / std).to(dev)
    maps, peaks = [], {}
    with torch.no_grad():
        for a in affs:
            out = model(x, gt_aff=torch.tensor([aff_names.index(a)], device=dev))
            hm = F.interpolate(out.unsqueeze(1), (H, W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            peaks[a] = float(hm.max())
            maps.append((hm - hm.min()) / (np.ptp(hm) + 1e-6))
    combined = np.maximum.reduce(maps)
    mask = (cv2.GaussianBlur((combined > mask_thresh).astype(np.float32), (7, 7), 0) > 0).astype(np.uint8)
    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilate)
    # keep only the blob with the highest mean heat (decision 2026-11-13); other blobs are dropped
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n > 2:
        best = max(range(1, n), key=lambda k: combined[labels == k].mean())
        mask = (labels == best).astype(np.uint8)
    peaks["blobs"] = n - 1
    full_mask, full_map = np.zeros((H_full, W_full), bool), np.zeros((H_full, W_full), np.float32)
    full_mask[y0c:y1c, x0c:x1c], full_map[y0c:y1c, x0c:x1c] = mask.astype(bool), combined
    return full_mask, full_map, peaks


def roi(mask, pad, shape):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    H, W = shape
    return max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)


def object_points(ref_img, ref_mask, ref_box, img, mask, box, a):
    """Paper: cross-DINO between the two ROIs, scale back to full frame, keep pairs inside both masks."""
    from d3_utils.correspondences import find_correspondences
    y0r, y1r, x0r, x1r = ref_box
    y0, y1, x0, x1 = box
    # The extractor flips the channels of a numpy input (it assumes BGR), so pass BGR to give DINO true RGB.
    pts_ref, pts_tgt, pil_ref, pil_tgt = find_correspondences(
        ref_img[y0r:y1r, x0r:x1r], img[y0:y1, x0:x1], num_pairs=a.pairs, load_size=a.load_size,
        layer=9, facet="key", bin=True, thresh=a.thresh, model_type="dino_vits8", stride=a.stride)
    sxr, syr = (x1r - x0r) / pil_ref.size[0], (y1r - y0r) / pil_ref.size[1]
    sx, sy = (x1 - x0) / pil_tgt.size[0], (y1 - y0) / pil_tgt.size[1]
    keep = []
    for (yr, xr), (yt, xt) in zip(pts_ref, pts_tgt):
        Yr, Xr, Yt, Xt = yr * syr + y0r, xr * sxr + x0r, yt * sy + y0, xt * sx + x0
        ir, jr, it, jt = int(round(Yr)), int(round(Xr)), int(round(Yt)), int(round(Xt))
        if (0 <= ir < ref_mask.shape[0] and 0 <= jr < ref_mask.shape[1] and 0 <= it < mask.shape[0]
                and 0 <= jt < mask.shape[1] and ref_mask[ir, jr] and mask[it, jt]):
            keep.append((Xt, Yt))
    return np.array(keep[:a.points], float).reshape(-1, 2), len(pts_ref), len(keep)


def track(cotracker, frames_bgr, pts, device):
    """CoTracker3 online, README pattern: queries on frame 0, then 8-frame steps over a 16-frame window."""
    video = torch.from_numpy(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]))  # T,H,W,3 uint8
    video = video.permute(0, 3, 1, 2)[None].float()                                                # 1,T,3,H,W 0-255
    T, step = video.shape[1], cotracker.step
    q = torch.cat([torch.zeros(len(pts), 1), torch.tensor(pts, dtype=torch.float32)], 1)[None].to(device)
    cotracker(video_chunk=video[:, :2 * step].to(device), is_first_step=True, queries=q)
    tracks = vis = None
    for ind in range(0, T - step, step):
        tracks, vis = cotracker(video_chunk=video[:, ind:ind + 2 * step].to(device))
    tracks, vis = tracks[0, :T].cpu().numpy(), vis[0, :T].cpu().numpy()
    assert tracks.shape[0] == T, f"CoTracker returned {tracks.shape[0]} of {T} frames"
    return tracks, vis


def load_frames(ep_dir):
    """frame numbers of robot_states.csv and their images from the raw recording (params.json "source")"""
    src = Path(json.loads((ep_dir / "params.json").read_text())["source"])
    rows = list(csv.DictReader(open(ep_dir / "robot_states.csv")))
    nums = [int(r["frame"]) for r in rows]
    return nums, [cv2.imread(str(src / f"{n}.png")) for n in nums]


def save_checks(out, frames, nums, mask, pts, tracks, vis, box, peaks, crop=None):
    f0 = frames[0].copy()
    over = f0.copy()
    over[mask] = (0.5 * over[mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    y0, y1, x0, x1 = box
    cv2.rectangle(over, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if crop:
        cv2.rectangle(over, tuple(crop[:2]), tuple(crop[2:]), (255, 0, 255), 1)
    cv2.putText(over, "OOAL " + ", ".join(f"{k} peak {v:.2f}" for k, v in peaks.items() if k != "blobs")
                + f"  (kept 1 of {peaks.get('blobs', 1)} blobs)", (20, 40), 0, 1, (0, 255, 255), 2)
    cv2.imwrite(str(out / "aff_mask_overlay.png"), over)
    p0 = f0.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(p0, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(p0, str(i + 1), (int(x) + 6, int(y) - 6), 0, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(out / "points_first_frame.png"), p0)
    # 8 frames, cropped around the points
    picks = np.linspace(0, len(frames) - 1, 8).astype(int)
    cx, cy = tracks[:, :, 0].mean(), tracks[:, :, 1].mean()
    x0c, y0c = int(np.clip(cx - 320, 0, 1280 - 640)), int(np.clip(cy - 200, 0, 720 - 400))
    tiles = []
    for k in picks:
        im = frames[k].copy()
        for i, (x, y) in enumerate(tracks[k]):
            cv2.circle(im, (int(x), int(y)), 4, (0, 0, 255) if vis[k, i] else (255, 0, 0), -1)
        im = im[y0c:y0c + 400, x0c:x0c + 640]
        cv2.putText(im, f"frame {nums[k]}  visible {int(vis[k].sum())}/{len(pts)}", (10, 30), 0, 0.8, (0, 255, 255), 2)
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out / "track_check.png"), cv2.resize(grid, None, fx=0.5, fy=0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/mug_a2a"), help="step-2 output folder (ep_*)")
    p.add_argument("--eps", nargs="*", help="which demos (default: all)")
    p.add_argument("--ref", help="reference demo (paper: the last one)")
    p.add_argument("--affs", nargs="+", default=["hold"])
    p.add_argument("--crop", type=int, nargs=4, default=[780, 160, 1120, 460], metavar=("X0", "Y0", "X1", "Y1"),
                   help="OOAL looks only inside this box (contains the mug in all 10 mug_a2a first frames, gripper "
                        "left out); 0 0 1280 720 = full frame as in the paper")
    p.add_argument("--mask_thresh", type=float, default=0.1)
    p.add_argument("--dilate", type=int, default=1)
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--points", type=int, default=15)
    p.add_argument("--pad", type=int, default=20)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--load_size", type=int, default=112)
    p.add_argument("--thresh", type=float, default=0.0, help="DINO saliency threshold")
    p.add_argument("--gpu", default="1", help="CUDA device (default 1: GPU 0 of the lab PC is shared)")
    a = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    device = "cuda"

    data = Path(a.data)
    all_eps = sorted(d.name for d in data.glob("ep_*") if (d / "robot_states.csv").exists())
    ref_name = a.ref or all_eps[-1]
    eps = a.eps or all_eps
    t0 = time.time()
    model, aff_names, (n_missing, n_unexp) = load_ooal(device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device).eval()
    print(f"models loaded ({time.time() - t0:.0f} s; OOAL missing/unexpected keys {n_missing}/{n_unexp})")

    _, ref_frames = load_frames(data / ref_name)
    ref_img = ref_frames[0]
    ref_mask, _, _ = affordance_mask(ref_img, model, aff_names, a.affs, a.mask_thresh, a.dilate, a.crop)
    ref_box = roi(ref_mask, a.pad, ref_img.shape[:2])

    for ep in eps:
        t1 = time.time()
        out = data / ep
        nums, frames = load_frames(out)
        mask, _, peaks = affordance_mask(frames[0], model, aff_names, a.affs, a.mask_thresh, a.dilate, a.crop)
        box = roi(mask, a.pad, frames[0].shape[:2])
        pts, n_pairs, n_both = object_points(ref_img, ref_mask, ref_box, frames[0], mask, box, a)
        if len(pts) == 0:
            print(f"{ep}: no points inside both masks -> skipped")
            continue
        with torch.no_grad():
            tracks, vis = track(cotracker, frames, pts, device)
        with open(out / "tracks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + sum([[f"obj_x{i + 1}", f"obj_y{i + 1}"] for i in range(a.points)], []))
            for k, n in enumerate(nums):
                row = [n]
                for i in range(a.points):
                    row += [f"{tracks[k, i, 0]:.2f}", f"{tracks[k, i, 1]:.2f}"] if i < len(pts) else [-1, -1]
                w.writerow(row)
        with open(out / "visibility.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + [f"vis{i + 1}" for i in range(a.points)])
            for k, n in enumerate(nums):
                w.writerow([n] + [int(vis[k, i]) if i < len(pts) else 0 for i in range(a.points)])
        save_checks(out, frames, nums, mask, pts, tracks, vis, box, peaks, a.crop)
        info = {**{k: v for k, v in vars(a).items() if k not in ("eps", "data", "gpu")}, "reference": ref_name,
                "ooal_peak": peaks, "mask_pixels": int(mask.sum()), "roi_y0y1x0x1": [int(v) for v in box],
                "dino_pairs": n_pairs, "pairs_in_both_masks": n_both, "points": len(pts),
                "visible_fraction": float(vis.mean()), "seconds": round(time.time() - t1, 1)}
        (out / "track_params.json").write_text(json.dumps(info, indent=2))
        print(f"{ep}: OOAL peak {list(peaks.values())[0]:.2f}, kept 1 of {peaks['blobs']} blobs, mask {int(mask.sum())} px, DINO {n_pairs} pairs, "
              f"{n_both} in both masks -> {len(pts)} points, visible {vis.mean():.0%}, {time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
