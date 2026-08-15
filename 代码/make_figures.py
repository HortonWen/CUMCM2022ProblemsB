# -*- coding: utf-8 -*-
"""Generate publication-quality figures for the paper."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

import solve_problemB as S

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG = S.FIG


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def fig_geometry():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    A = np.array([-2.0, 0.0])
    B = np.array([2.0, 0.0])
    P = np.array([0.0, 2.4])
    ax.plot([P[0], A[0]], [P[1], A[1]], color="0.25", lw=1.2)
    ax.plot([P[0], B[0]], [P[1], B[1]], color="0.25", lw=1.2)
    ax.plot([A[0], B[0]], [A[1], B[1]], color="0.75", lw=1.0, ls="--")
    ax.scatter(*A, s=70, color="#d62728", zorder=3)
    ax.scatter(*B, s=70, color="#d62728", zorder=3)
    ax.scatter(*P, s=90, color="#1f77b4", zorder=3)
    ax.annotate("发射机 A", A + [0.08, -0.42], fontsize=10)
    ax.annotate("发射机 B", B + [0.08, -0.42], fontsize=10)
    ax.annotate("接收机 P", P + [0.08, 0.12], fontsize=10)
    # angle arc at P
    arc = Arc(P, 1.5, 1.5, angle=0, theta1=-65, theta2=-115, color="black", lw=1.0)
    ax.add_patch(arc)
    ax.text(0.05, 1.62, r"$\alpha$", fontsize=12)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.1, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("纯方位被动定位的方向信息：接收机与两发射机连线夹角", fontsize=11)
    save(fig, "fig1_geometry.png")


def fig_circle_formation():
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(S.R * np.cos(theta), S.R * np.sin(theta), color="0.85", lw=1.0, zorder=1)
    for j in S.CIRCLE_IDS:
        init = S.INITIAL[j]
        ideal = S.IDEAL[j]
        ax.plot([init[0], ideal[0]], [init[1], ideal[1]], color="#ff7f0e", lw=0.8, alpha=0.7)
        ax.scatter(*ideal, s=70, color="#1f77b4", zorder=3)
        ax.scatter(*init, s=42, color="#d62728", zorder=4)
        ax.annotate(f"FY{j:02d}", ideal + np.array([3.5, 3.5]), fontsize=8, color="#1f77b4")
    ax.scatter(*S.IDEAL[0], s=80, marker="*", color="black", zorder=5)
    ax.annotate("FY00", S.IDEAL[0] + np.array([3.0, -9.0]), fontsize=9)
    ax.plot([], [], "o", color="#d62728", label="初始位置")
    ax.plot([], [], "o", color="#1f77b4", label="理想位置")
    ax.plot([], [], color="#ff7f0e", lw=1.0, label="调整位移")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("圆形编队：初始位置与理想位置（R=100 m）")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    save(fig, "fig2_circle_formation.png")


def fig_adjustment_rounds():
    # Simulate the round positions without relying on the JSON file.
    rounds = S.q1_3_rounds(verbose=False)["rounds"]
    pos = {j: S.INITIAL[j].copy() for j in S.INITIAL}
    snapshots = [dict(pos)]
    for rd in rounds:
        for mv in rd["movements"]:
            j = mv["id"]
            pos[j] = np.array(mv["target_xy"], dtype=float)
        snapshots.append(dict(pos))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    titles = ["初始状态", "第 1 轮调整后", "第 2 轮调整后（完成）"]
    for k, ax in enumerate(axes):
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(S.R * np.cos(theta), S.R * np.sin(theta), color="0.85", lw=1.0)
        snap = snapshots[k]
        for j in S.CIRCLE_IDS:
            ideal = S.IDEAL[j]
            cur = snap[j]
            ax.plot([cur[0], ideal[0]], [cur[1], ideal[1]], color="#ff7f0e", lw=0.7, alpha=0.7)
            ax.scatter(*ideal, s=52, color="#1f77b4", zorder=3)
            ax.scatter(*cur, s=34, color="#d62728", zorder=4)
        ax.scatter(*S.IDEAL[0], s=70, marker="*", color="black", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(titles[k], fontsize=10)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(alpha=0.22)
    fig.suptitle("问题 1(3) 迭代调整过程", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig3_adjustment_rounds.png")


def fig_cone():
    ideal = S.cone_ideal()
    # Use the deterministic initial deviations as generated in solve_q2.
    rng = np.random.default_rng(2022)
    init = [p + rng.uniform(-3.0, 3.0, size=2) for p in ideal]
    init[0] = ideal[0].copy()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # Draw formation links among ideal points.
    links = [(0, 1), (0, 2), (1, 3), (2, 5), (3, 4), (4, 5)]
    for a, b in links:
        ax.plot([ideal[a][0], ideal[b][0]], [ideal[a][1], ideal[b][1]],
                color="0.85", lw=1.2, zorder=1)
    for i, p in enumerate(ideal):
        q = init[i]
        ax.plot([q[0], p[0]], [q[1], p[1]], color="#ff7f0e", lw=0.9, alpha=0.8)
        ax.scatter(*p, s=68, color="#1f77b4", zorder=3)
        ax.scatter(*q, s=42, color="#d62728", zorder=4)
        ax.annotate(f"{i}", p + np.array([2.2, 2.2]), fontsize=8, color="#1f77b4")
    ax.plot([], [], "o", color="#d62728", label="初始位置")
    ax.plot([], [], "o", color="#1f77b4", label="理想模板位置")
    ax.plot([], [], color="#ff7f0e", lw=1.0, label="调整位移")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("问题 2：锥形编队模板与位置调整")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    save(fig, "fig4_cone.png")


if __name__ == "__main__":
    fig_geometry()
    fig_circle_formation()
    fig_adjustment_rounds()
    fig_cone()
