"""Step 6 (paper: data_generation/make_reference.py): the reference a robot run matches its first frame against.

    python 6_make_reference.py --data data/mug_a2a --ref ep_009       # = the --ref used in 3_track.py

Same processing as 3_track.py (OOAL inside the box, strongest blob, ROI), so a robot run picks its 15 points the same
way the training demos did. Writes references/<task>/: ref_image.png, ref_mask.png, ref_overlay.png, params.json.
Needs the raw recording of the reference demo (its first image). references/mug_a2a is included in this repo.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from common import HERE

spec = importlib.util.spec_from_file_location("track", HERE / "3_track.py")
track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(track)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/mug_a2a"))
    p.add_argument("--ref", default="ep_009", help="the reference demo used by 3_track.py")
    p.add_argument("--out", default=str(HERE / "references/mug_a2a"))
    a = p.parse_args()
    ep = Path(a.data) / a.ref
    tp = json.loads((ep / "track_params.json").read_text())  # settings step 3 used
    nums, frames = track.load_frames(ep)
    img = frames[0]
    model, names, _ = track.load_ooal("cuda")
    mask, _, peaks = track.affordance_mask(img, model, names, tp["affs"], tp["mask_thresh"], tp["dilate"], tp["crop"])
    box = track.roi(mask, tp["pad"], img.shape[:2])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "ref_image.png"), img)
    cv2.imwrite(str(out / "ref_mask.png"), mask.astype(np.uint8) * 255)
    over = img.copy()
    over[mask] = (0.5 * over[mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    cv2.rectangle(over, (box[2], box[0]), (box[3], box[1]), (0, 255, 0), 2)
    cv2.imwrite(str(out / "ref_overlay.png"), over)
    keep = ("affs", "mask_thresh", "dilate", "crop", "pairs", "points", "pad", "stride", "load_size", "thresh")
    info = {"source": f"{a.ref} frame {nums[0]}", **{k: tp[k] for k in keep},
            "roi_y0y1x0x1": [int(v) for v in box], "ooal_peak": peaks}
    (out / "params.json").write_text(json.dumps(info, indent=2))
    print(f"reference -> {out} (mask {int(mask.sum())} px, ROI {box})")


if __name__ == "__main__":
    main()
