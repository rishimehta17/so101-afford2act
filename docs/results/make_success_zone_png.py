"""docs/success_zone.png from docs/results/rollouts.csv (all rollouts that closed the gripper, 2026-11-16/17)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
R = pd.read_csv(HERE / "rollouts.csv")
c = R[R.t_close.notna()].copy()
kind = {"lifted": "lifted the mug", "lifted, dropped": "lifted the mug", "hooked / tipped": "got the handle, no clean lift",
        "closed on handle, no time": "got the handle, no clean lift"}
c["kind"] = c.outcome.map(kind).fillna("missed")
c["x"] = c.dev_px.clip(upper=130)
zone = (c.dev_px <= 20) & (c.tip_z_close <= 32)
got = c.kind != "missed"
BLUE, GREY, INK, INK2, SURF = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURF)
ax.set_facecolor(SURF)
ax.add_patch(plt.Rectangle((0, -10), 20, 42, color=BLUE, alpha=0.08, lw=0))
ax.add_patch(plt.Rectangle((0, -10), 20, 42, fill=False, ec=BLUE, lw=1))
ax.text(1.5, -7, "success zone", color=INK, fontsize=10, fontweight="bold")
ax.axhline(33, xmin=20 / 130, color=GREY, ls="--", lw=1)
ax.text(129, 34.5, "demos grasp at 33 mm (median)", color=INK2, fontsize=9, ha="right")
m = c[c.kind == "missed"]
ax.scatter(m.x, m.tip_z_close, s=28, color=GREY, alpha=0.6, lw=0, label=f"missed ({len(m)})")
h = c[c.kind == "got the handle, no clean lift"]
ax.scatter(h.x, h.tip_z_close, s=60, facecolors="none", edgecolors=BLUE, lw=2, label=f"got the handle, no clean lift ({len(h)})")
l_ = c[c.kind == "lifted the mug"]
ax.scatter(l_.x, l_.tip_z_close, s=70, color=BLUE, label=f"lifted the mug ({len(l_)})")
ax.set_xlim(0, 132), ax.set_ylim(-10, 100)
ax.set_xlabel("distance from the demos' grasp spot in the image at the close (px; >130 shown at 130)", color=INK2)
ax.set_ylabel("gripper tip height at the close (mm)", color=INK2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GREY)
ax.tick_params(colors=INK2)
ax.grid(axis="y", color="#e6e5e1", lw=0.8)
ax.set_axisbelow(True)
fig.suptitle("Grasps work when the gripper closes low and on the spot", x=0.02, ha="left", fontsize=14,
             fontweight="bold", color=INK)
ax.set_title(f"{got[zone].sum()} of {zone.sum()} closes inside the zone got the handle; "
             f"{got[~zone].sum()} of {(~zone).sum()} outside it did ({len(c)} closes, 58 rollouts)",
             loc="left", fontsize=10, color=INK2)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(HERE.parent / "success_zone.png", dpi=110, facecolor=SURF)
print("->", HERE.parent / "success_zone.png")
