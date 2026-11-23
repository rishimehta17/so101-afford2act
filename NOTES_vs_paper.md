# What differs from the Afford2Act paper, and why

The perception and learning core is the paper's: the 15 handle points (OOAL "hold" + DINO matching on the first frame,
CoTracker afterwards) and the policy network (TinyTransformer with a gate, 10-step action chunks, temporal ensembling,
MSE + BCE, 1000 epochs). Everything below was changed because it failed on the SO101; each change was measured before
and after on the robot.

| Part | Paper (UR3e) | This repo (SO101) | Why |
|---|---|---|---|
| Gripper points | ArUco marker (authors' code) | 4 points computed from the measured joints: forward kinematics + a camera calibration fitted from the demos (`fk_points.py`, `4_fit_camera.py`) | Near the grasp the marker was seen in only 50-69 % of frames and the box tracker drifted by up to 330 px. The computed points are never hidden and land within 2.5-3.4 px of the real marker on demos the fit never saw |
| Time input | none in the paper's text; frame number in the authors' code | none | With the frame number the policy closed the gripper at about 11 s wherever it was, then played "put down" with an empty gripper |
| Input frames | 1 frame, 19 tokens | now, 0.5 s and 1.0 s ago: 57 tokens (`--history 0 5 10`) | Without a clock, "approaching" and "lifting" look alike; motion over the last second tells them apart (the authors' code has this option, `KP_WINDOW`, set to 1) |
| Task trained on | grasp-and-lift | the grasp-and-lift part of our longer demos (`--segment grasp_lift`) | Our demos also put the mug down and return home: look-alike moments the policy cannot separate without time |
| Action | gripper-pose change (6 numbers), executed with IK | change of the 5 arm joints (deg) | On the 5-joint SO101, pose steps through IK were lost in servo slack (the arm moved 0-7 % of what was asked) |
| Next target | measured pose + change | previous target + change, kept within 10 deg of the measured joints | Same reason: small steps from the measured pose disappeared |
| Execution | stream commands, never wait | feedback loop: wait until every joint is within 1 deg (max 0.4 s), plus a gravity-sag offset | The SO101 trailed each command by 4-8 mm (worst 10 %: 10-15 mm); with the loop 1.5-1.8 mm |
| Gripper | closed if P(closed) > 0.5 | opened before step 1; P(closed) averaged over the last 10 plans, close above 0.6, open below 0.3 only after 2 s closed | The policy waited for an open gripper before moving (0-14 s, or never), and near the handle 13 of 31 closes were < 1 s flickers |

Fixes to the paper's code that do not change the method (`3_track.py`):
- OOAL input squashed to 224x224 (OOAL's own test transform) instead of resize + centre crop, which shifted the heatmap.
- DINO gets true RGB (the paper's call path swaps red and blue).
- CoTracker gets 0-255 RGB frames (the paper's wrapper divided by 255 twice) and is called as in its README.
- OOAL looks only inside a fixed box around the mug area: on the full frame the handle was smaller than one OOAL patch.
- Frame numbers of tracks and gripper points are the same image numbers (the paper's files were off by one).
- Training windows stay inside one demo; validation uses whole held-out demos; scalers come from training demos only.

What is still open (from the 58 analysed rollouts, `docs/results/`):
- The policy sees only 2D image points, and from one camera "higher" and "further back" look almost the same, so its
  height at the close varies from about -7 to 95 mm. Grasps work when the close lands within 20 px of the demos' grasp
  spot and at most 32 mm high (`docs/success_zone.png`).
- After the grasp, CoTracker loses about half of the handle points (the gripper covers them).
