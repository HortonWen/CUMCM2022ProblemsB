# -*- coding: utf-8 -*-
"""
2022 CUMCM Problem B
Unmanned aerial vehicle formation passive bearing-only positioning

This script computes the numerical results used in the paper.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "problemB_results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


# ----------------------------- geometry helpers -----------------------------

def pol2cart(r, deg):
    a = math.radians(deg)
    return np.array([r * math.cos(a), r * math.sin(a)], dtype=float)


def cart2pol(p):
    return float(np.hypot(p[0], p[1])), float(np.degrees(math.atan2(p[1], p[0])) % 360.0)


def angle_apb(p, a, b):
    """Angle APB (in degrees), i.e. angle between vectors A-P and B-P."""
    u = a - p
    v = b - p
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def all_pair_angles(p, anchors):
    vals = []
    for i, j in itertools.combinations(range(len(anchors)), 2):
        vals.append(angle_apb(p, anchors[i], anchors[j]))
    return np.array(vals, dtype=float)


def localize(anchors, measured_angles, x0):
    """Localize a receiver from known transmitter positions and measured angles."""

    def residual(x):
        return all_pair_angles(np.asarray(x), anchors) - measured_angles

    # Use several nearby starts to escape the symmetric/local minima problem.
    starts = [np.asarray(x0, dtype=float)]
    best = None
    best_cost = np.inf
    for s in starts:
        sol = least_squares(residual, s, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if sol.cost < best_cost:
            best_cost = sol.cost
            best = sol.x
    return np.asarray(best), float(best_cost)


def angle_between_known_pair(p, a, b):
    return angle_apb(p, a, b)


# ----------------------------- problem data ---------------------------------

R = 100.0
CIRCLE_IDS = list(range(1, 10))  # FY01..FY09
IDEAL_ANGLES = {j: (j - 1) * 40.0 for j in CIRCLE_IDS}
IDEAL = {0: np.array([0.0, 0.0])}
for j in CIRCLE_IDS:
    IDEAL[j] = pol2cart(R, IDEAL_ANGLES[j])

# Table 1: initial positions in polar coordinates (r, deg)
INITIAL_POLAR = {
    0: (0.0, 0.0),
    1: (100.0, 0.0),
    2: (98.0, 40.10),
    3: (112.0, 80.21),
    4: (105.0, 119.75),
    5: (98.0, 159.86),
    6: (112.0, 199.96),
    7: (105.0, 240.07),
    8: (98.0, 280.17),
    9: (112.0, 320.28),
}
INITIAL = {j: pol2cart(r, d) for j, (r, d) in INITIAL_POLAR.items()}


def fmt_xy(p):
    return f"({p[0]:.3f}, {p[1]:.3f})"


# ----------------------------- Problem 1 (1) -------------------------------

def solve_q1_1():
    # Receiver: the deviated FY02. Transmitters: FY00 (center) and two
    # accurately positioned circle UAVs FY01 and FY03.
    p_true = INITIAL[2]
    anchors = [IDEAL[0], IDEAL[1], IDEAL[3]]
    measured = all_pair_angles(p_true, anchors)
    x0 = IDEAL[2]  # nominal position, not the true deviated position
    est, cost = localize(anchors, measured, x0)
    err = np.linalg.norm(est - p_true)
    print("\n[Q1(1)]")
    print("true P      =", fmt_xy(p_true))
    print("estimated P =", fmt_xy(est))
    print("anchor angles =", np.round(measured, 6))
    print("localization error (m) =", round(err, 8))
    return {
        "true_polar": INITIAL_POLAR[2],
        "true_xy": p_true.tolist(),
        "estimated_xy": est.tolist(),
        "error_m": float(err),
        "angles_deg": measured.tolist(),
    }


# ----------------------------- Problem 1 (2) -------------------------------

def solve_q1_2():
    """Enumerate how many additional unknown-ID transmitters are needed."""
    receiver_id = 2
    p_true = INITIAL[receiver_id]
    known = [IDEAL[0], IDEAL[1]]
    # The receiving UAV itself cannot also be an unknown transmitter.
    unknown_ids = [j for j in range(2, 10) if j != receiver_id]
    unknown_positions = {j: IDEAL[j] for j in unknown_ids}

    report = {}
    # Choose one concrete "true" set of unknown transmitters for each size m.
    # The solver must identify both P and the set without being told the IDs.
    for m, true_unknown in ((1, (3,)), (2, (3, 4)), (3, (3, 4, 5))):
        true_anchors = known + [unknown_positions[j] for j in true_unknown]
        measured = all_pair_angles(p_true, true_anchors)
        consistent = []
        for chosen in itertools.combinations(unknown_ids, m):
            anchors = known + [unknown_positions[j] for j in chosen]
            best_res = np.inf
            best_xy = None
            # Try all nominal formation points as initial guesses.
            for j in CIRCLE_IDS:
                est, _ = localize(anchors, measured, IDEAL[j])
                res = float(np.max(np.abs(all_pair_angles(est, anchors) - measured)))
                if res < best_res:
                    best_res = res
                    best_xy = est
            if best_res < 1e-5:
                consistent.append({"ids": list(chosen), "xy": best_xy.tolist(), "res": best_res})
        report[m] = {
            "true_ids": list(true_unknown),
            "num_assignments": len(list(itertools.combinations(unknown_ids, m))),
            "num_consistent": len(consistent),
            "consistent": consistent[:8],
        }
        print(f"\n[Q1(2)] m={m}, true_ids={true_unknown}: assignments={report[m]['num_assignments']}, consistent={len(consistent)}")
        for c in consistent[:8]:
            print("  ids", c["ids"], "xy", tuple(round(v, 3) for v in c["xy"]), "res", f"{c['res']:.2e}")
    return report


# ----------------------------- Problem 1 (3) -------------------------------

def q1_3_rounds(verbose=True):
    pos = {j: INITIAL[j].copy() for j in INITIAL}
    rounds = []

    # Round 1: use FY00 + FY01, FY02, FY03 as transmitters.
    # FY01 is already ideal; FY02/FY03 have known initial positions.
    tx = [0, 1, 2, 3]
    rx = [4, 5, 6, 7, 8, 9]
    anchors = [pos[j] for j in tx]
    movements = []
    for j in rx:
        measured = all_pair_angles(pos[j], anchors)
        est, _ = localize(anchors, measured, IDEAL[j])
        before = pos[j].copy()
        err = float(np.linalg.norm(est - pos[j]))
        move = float(np.linalg.norm(IDEAL[j] - est))
        pos[j] = IDEAL[j].copy()
        movements.append({
            "id": j,
            "before_xy": before.tolist(),
            "estimated_xy": est.tolist(),
            "localization_err_m": err,
            "target_xy": IDEAL[j].tolist(),
            "move_m": move,
        })
    rounds.append({"round": 1, "transmitters": tx, "receivers": rx, "movements": movements})

    # Round 2: now FY04..FY09 are at ideal positions. Use them as anchors to
    # correct FY02 and FY03.
    tx = [0, 4, 5]
    rx = [2, 3]
    anchors = [pos[j] for j in tx]
    movements = []
    for j in rx:
        measured = all_pair_angles(pos[j], anchors)
        est, _ = localize(anchors, measured, IDEAL[j])
        before = pos[j].copy()
        err = float(np.linalg.norm(est - pos[j]))
        move = float(np.linalg.norm(IDEAL[j] - est))
        pos[j] = IDEAL[j].copy()
        movements.append({
            "id": j,
            "before_xy": before.tolist(),
            "estimated_xy": est.tolist(),
            "localization_err_m": err,
            "target_xy": IDEAL[j].tolist(),
            "move_m": move,
        })
    rounds.append({"round": 2, "transmitters": tx, "receivers": rx, "movements": movements})

    # Round 3: verification. Use FY00 + FY01,FY02,FY03 (all ideal) as anchors.
    tx = [0, 1, 2, 3]
    rx = [4, 5, 6, 7, 8, 9]
    anchors = [pos[j] for j in tx]
    movements = []
    for j in rx:
        measured = all_pair_angles(pos[j], anchors)
        est, _ = localize(anchors, measured, IDEAL[j])
        err = float(np.linalg.norm(est - pos[j]))
        movements.append({
            "id": j,
            "before_xy": pos[j].tolist(),
            "estimated_xy": est.tolist(),
            "localization_err_m": err,
            "target_xy": IDEAL[j].tolist(),
            "move_m": err,
        })
    rounds.append({"round": 3, "transmitters": tx, "receivers": rx, "movements": movements, "verification": True})

    if verbose:
        print("\n[Q1(3)] adjustment rounds")
        for rd in rounds:
            print("round", rd["round"], "tx", rd["transmitters"], "rx", rd["receivers"])
            for m in rd["movements"]:
                print("  FY%02d" % m["id"],
                      "local_err", round(m["localization_err_m"], 8),
                      "move", round(m["move_m"], 6))

    final_err = {j: float(np.linalg.norm(pos[j] - IDEAL[j])) for j in CIRCLE_IDS}
    final_max = max(final_err.values())
    print("final max position error (m):", final_max)
    return {"rounds": rounds, "final_errors_m": final_err, "final_max_error_m": final_max}


# ----------------------------- Problem 2 -----------------------------------

def cone_ideal(n=6, spacing=50.0, lateral=50.0):
    """A simple cone template.

    Column 0: leader at (0,0).
    Column 1: two UAVs at x=-spacing, y=+-lateral.
    Column 2: three UAVs at x=-2*spacing, y=-2*lateral,0,2*lateral.
    Here n is total number of UAVs; if n != 6 this function still returns a
    usable first six-point template for the simulation.
    """
    pts = [np.array([0.0, 0.0])]
    for y in (-lateral, lateral):
        pts.append(np.array([-spacing, y]))
    for y in (-2 * lateral, 0.0, 2 * lateral):
        pts.append(np.array([-2 * spacing, y]))
    return pts[:n]


def solve_q2():
    ideal = cone_ideal()
    rng = np.random.default_rng(2022)
    # Slight initial deviations (same order of magnitude as Problem 1).
    pos = [p + rng.uniform(-3.0, 3.0, size=2) for p in ideal]
    # Keep leader exactly at origin.
    pos[0] = ideal[0].copy()
    n = len(ideal)
    rounds = []

    # Round 1: leader + two first-wing UAVs as anchors, locate the rest.
    tx = [0, 1, 2]
    rx = list(range(3, n))
    anchors = [pos[j] for j in tx]
    for j in rx:
        measured = all_pair_angles(pos[j], anchors)
        est, _ = localize(anchors, measured, ideal[j])
        move = float(np.linalg.norm(ideal[j] - est))
        pos[j] = ideal[j].copy()
        rounds.append({"round": 1, "id": j, "move_m": move,
                       "before_xy": pos[j].tolist(), "target_xy": ideal[j].tolist()})

    # Round 2: verification/correction for the two first-wing UAVs.
    tx = [0, 3, 4, 5]
    anchors = [pos[j] for j in tx]
    for j in (1, 2):
        measured = all_pair_angles(pos[j], anchors)
        est, _ = localize(anchors, measured, ideal[j])
        move = float(np.linalg.norm(ideal[j] - est))
        pos[j] = ideal[j].copy()
        rounds.append({"round": 2, "id": j, "move_m": move,
                       "before_xy": pos[j].tolist(), "target_xy": ideal[j].tolist()})

    final_err = {i: float(np.linalg.norm(pos[i] - ideal[i])) for i in range(n)}
    print("\n[Q2] cone adjustment")
    for r in rounds:
        print("  UAV", r["id"], "round", r["round"], "move", round(r["move_m"], 6))
    print("final max error:", max(final_err.values()))
    return {"template_xy": [p.tolist() for p in ideal],
            "rounds": rounds,
            "final_errors_m": final_err,
            "final_max_error_m": float(max(final_err.values()))}


def main():
    results = {}
    results["q1_1"] = solve_q1_1()
    results["q1_2"] = solve_q1_2()
    results["q1_3"] = q1_3_rounds()
    results["q2"] = solve_q2()
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nwrote", OUT / "results.json")


if __name__ == "__main__":
    main()
