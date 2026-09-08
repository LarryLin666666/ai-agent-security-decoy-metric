"""Generate all Working Note figures as PNG for Kaggle-writeup embedding.
Palette: validated dataviz-skill default (light mode, categorical order fixed).
Run: python make_figures.py  (writes ./*.png next to this script)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- palette (references/palette.md) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # slot 1 - categorical primary / "public"
ORANGE = "#eb6834"    # slot 2 - "private"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
SUCCESS_TXT = "#006300"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "font.size": 12,
})


def _clean(ax, left=True, bottom=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.array([0, 2000])
    y = 0.09 * x
    ax.plot(x, y, color=BLUE, lw=2.4, zorder=3)
    ax.axhline(180, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.axvline(2000, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(1960, 186, "replay cap  N=2000 → 180", ha="right", color=RED, fontsize=10.5)
    ax.scatter([995], [89.6], s=48, color=BLUE, zorder=4)
    ax.annotate("89.6 · ours (~995)", (995, 89.6), xytext=(1060, 74),
                fontsize=11, color=INK)
    ax.scatter([1530], [138], s=48, color=BLUE, zorder=4)
    ax.annotate("138 · public top (~1530)", (1530, 138), xytext=(1180, 148),
                fontsize=11, color=INK)
    ax.set_xlim(0, 2050)
    ax.set_ylim(0, 200)
    ax.set_xlabel("effective candidates  N$_{eff}$")
    ax.set_ylabel("normalized score")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.set_title("Score is a straight line through the origin", loc="left",
                  fontsize=13.5, color=INK, pad=12, fontweight="bold")
    save(fig, "fig1_score_vs_neff.png")


# ---------------------------------------------------------------- Fig 2
def fig2():
    rows = [
        ("aggressive fill 0.99", 87.3, "flat"),
        ("shorter message", 85.3, "flat"),
        ("default (variance)", 84.4, "flat"),
        ("multi-post ×5", 80.7, "down"),
        ("multi-post ×8", 78.1, "down"),
        ("blind-emit (no probe)", 53.0, "down"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [MUTED if r[2] == "flat" else RED for r in rows][::-1]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.56, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=10.5, color=INK)
    ax.axvline(89.55, color=BLUE, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(91.5, 1.5, "single-post\nbaseline\n89.6", color=BLUE,
            ha="left", fontsize=9.5, va="center", linespacing=1.35)
    ax.set_xlim(0, 108)
    ax.annotate("terse phrasing: fires 1/6, effectively 0\n(off this axis)", xy=(2, -0.9),
                fontsize=9.5, color=RED, ha="left", annotation_clip=False)
    ax.set_yticks(y, labels)
    ax.set_xlabel("public score")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax, left=False)
    ax.tick_params(left=False)
    ax.set_title("Refuted levers: nothing beats the single-post baseline", loc="left",
                  fontsize=13.5, color=INK, pad=14, fontweight="bold")
    save(fig, "fig2_refuted_levers.png")


# ---------------------------------------------------------------- Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    cats = ["baseline", "shorter"]
    x = np.arange(2)
    w = 0.5

    ax = axes[0]
    vals = [1.14, 0.95]
    ax.bar(x, vals, width=w, color=[MUTED, BLUE], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}s", ha="center", fontsize=10.5, color=INK)
    ax.text(1, 1.22, "−17% faster", color=SUCCESS_TXT, ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("local wall-clock / candidate (s)")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    vals = [84.4, 85.3]
    ax.bar(x, vals, width=w, color=[MUTED, MUTED], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=10.5, color=INK)
    ax.text(1, 96, "≈ unchanged", color=INK2, ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 100)
    ax.set_ylabel("evaluator public score")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("A 17% local speedup produced no evaluator change", x=0.02, ha="left",
                  fontsize=13.5, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_harness_fidelity.png")


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    k = np.array([1, 2, 3, 5])
    ideal = k
    gemma = np.array([1, 2, 3, 5])
    gpt = np.array([1, 2, 3, 4.2])
    ax.plot(k, gemma, color=AQUA, lw=2.2, marker="o", ms=6, zorder=4,
            label="gemma  (5/5 @ K=5 — matches the ideal exactly)")
    ax.plot(k, gpt, color=BLUE, lw=2.2, marker="o", ms=6, zorder=3, label="gpt-oss  (4.2/5 @ K=5)")
    ax.set_xticks(k, [f"K={v}" for v in k])
    ax.set_ylim(0, 6)
    ax.set_ylabel("writes credited / candidate")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Repeated destructive-write stacks linearly", loc="left", fontsize=12.5,
                  color=INK, fontweight="bold", pad=10)

    ax = axes[1]
    ks = ["K=3", "K=5", "K=8"]
    rates = [58, 35, 17]
    xb = np.arange(3)
    ax.bar(xb, rates, width=0.55, color=RED, zorder=3)
    for xi, v in zip(xb, rates):
        ax.text(xi, v + 2, f"{v}%", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(xb, ks)
    ax.set_ylim(0, 75)
    ax.set_ylabel("EXFIL fire-rate")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Contrast: multi-target exfil collapses", loc="left", fontsize=12.5,
                  color=CRITICAL, fontweight="bold", pad=10)

    fig.tight_layout()
    save(fig, "fig4_destructive_stack.png")


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    cols = ["plain text", "Morse", "base64", "ROT13"]
    rows = ["gpt-oss", "Gemma 4"]
    # 1 = fires/vulnerable, 0 = safe
    grid = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    for i in range(2):
        for j in range(4):
            fires = grid[i, j]
            color = CRITICAL if fires else "#eceae4"
            txt_color = "white" if fires else MUTED
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.92, 0.92, facecolor=color,
                                        edgecolor=SURFACE, linewidth=3))
            label = "EXECUTES" if fires else "reads only"
            sub = "exfiltrates" if (fires and i == 0) else ("runs shell" if fires else "")
            ax.text(j + 0.46, 1 - i + 0.56, label, ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")
            if sub:
                ax.text(j + 0.46, 1 - i + 0.32, sub, ha="center", va="center",
                        fontsize=8.5, color=txt_color)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.2)
    ax.set_xticks([0.46, 1.46, 2.46, 3.46], cols, fontsize=11, color=INK2)
    ax.set_yticks([1.46, 0.46], rows, fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Does the model decode & execute the injected instruction?", loc="left",
                  fontsize=12.5, color=INK, fontweight="bold", pad=12)
    save(fig, "fig5_injection_matrix.png")


# ---------------------------------------------------------------- Fig 7
def fig7():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    k = [1, 2, 4, 6]
    vals = [4.5, 6.8, 9.4, 10.8]
    x = np.arange(len(k))
    colors = [MUTED, MUTED, BLUE, GOOD]
    ax.bar(x, vals, width=0.55, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=11, color=INK)
    ax.axhline(9.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, 9.55, "marker's own raw/gen · 9.0", color=INK2, fontsize=10, ha="center")
    ax.set_xticks(x, [f"K={v}" for v in k])
    ax.set_ylim(0, 13)
    ax.set_ylabel("raw / generation")
    ax.set_xlabel("posts amortized over one read (both target models)")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Read-once, post-K beats the marker's own throughput", loc="left",
                  fontsize=13.5, color=INK, fontweight="bold", pad=12)
    save(fig, "fig7_readpost_throughput.png")


# ---------------------------------------------------------------- NEW: public vs private
def fig8():
    fams = ["Confused-\ndeputy", "Marker\n(EXFIL)", "Window-\nevasion", "Read-and-\npost", "Destructive-\nwrite K=1"]
    pub = [26.35, 89.55, 88.88, 0.00, 19.83]
    priv = [26.70, 0.00, 0.00, 0.00, 20.01]
    x = np.arange(len(fams))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar(x - w / 2, pub, width=w, color=BLUE, zorder=3, label="public")
    b2 = ax.bar(x + w / 2, priv, width=w, color=ORANGE, zorder=3, label="private (revealed)")
    for xi, v in zip(x - w / 2, pub):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
    for xi, v in zip(x + w / 2, priv):
        col = SUCCESS_TXT if v > 5 else CRITICAL
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_xticks(x, fams, fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("score")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.set_title("Public predicted almost nothing about private", loc="left",
                  fontsize=14, color=INK, fontweight="bold", pad=14)
    save(fig, "fig8_public_vs_private.png")


# ---------------------------------------------------------------- NEW: external replication
def fig9():
    # Ordered by private score. All seven are CONFUSED_DEPUTY-predicate scores; the spread
    # (26.7 -> 46.5) is set by throughput/technique, not by a different predicate. #1 (xz)
    # and #4 (Kohei/Rick) added post-close from the actual 1st/4th-place writeups.
    teams = ["ours\n(conf_forge)", "18th\n(Arvai)", "15th\n(Rice)", "13th\n(Kim)",
             "7th\n(Civitasmass)", "4th\n(Kohei)", "1st\n(xz)"]
    vals = [26.70, 29.23, 29.52, 29.67, 35.72, 41.325, 46.5]
    colors = [BLUE, MUTED, MUTED, MUTED, MUTED, MUTED, GOOD]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(teams))
    ax.bar(x, vals, width=0.60, color=colors, zorder=3)
    for xi, v, c in zip(x, vals, colors):
        ax.text(xi, v + 0.6, f"{v:.2f}", ha="center", fontsize=10.5, color=INK,
                fontweight=("bold" if c != MUTED else "normal"))
    ax.set_xticks(x, teams, fontsize=9.5)
    ax.set_ylim(0, 52)
    ax.set_ylabel("private score")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Seven independent teams, one surviving predicate — technique sets the spread",
                 loc="left", fontsize=12.4, color=INK, fontweight="bold", pad=12)
    save(fig, "fig9_external_replication.png")


# ---------------------------------------------------------------- NEW: field outcome
def fig10():
    fig, ax = plt.subplots(figsize=(7.6, 1.9))
    zero_pct = 2855 / 4252 * 100
    rest_pct = 100 - zero_pct
    ax.barh([0], [zero_pct], color=CRITICAL, height=0.5, zorder=3, label="private = 0.000")
    ax.barh([0], [rest_pct], left=[zero_pct], color=GOOD, height=0.5, zorder=3, label="private > 0")
    ax.text(zero_pct / 2, 0, f"{zero_pct:.0f}%\n2,855 teams", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    ax.text(zero_pct + rest_pct / 2, 0, f"{rest_pct:.0f}%\n1,397 teams", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("share of 4,252 teams, final private leaderboard")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title("Two-thirds of the field scored zero on the board that mattered", loc="left",
                  fontsize=12.8, color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "fig10_field_outcome.png")


# ---------------------------------------------------------------- NEW: rank curve
def fig11():
    # top-49 private scores scraped from the real leaderboard (rank: score)
    scores = [46.425,42.990,42.175,41.325,40.365,37.690,34.510,33.300,30.860,30.810,
              30.040,29.745,29.670,29.625,29.520,29.460,29.355,29.230,29.070,29.055,
              29.040,29.040,29.040,29.010,28.905,28.860,28.845,28.785,28.650,28.620,
              28.545,28.410,28.020,27.630,27.625,27.300,27.270,27.225,27.030,26.965,
              26.850,26.520,26.415,26.290,26.205,26.175,26.130,26.130,26.115]
    ranks = np.arange(1, len(scores) + 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(ranks, scores, color=MUTED, lw=2, zorder=2)
    ax.fill_between(ranks, scores, color=MUTED, alpha=0.08, zorder=1)
    # our score would slot in around rank 40-41
    our_rank = 41
    our_score = 26.695
    ax.scatter([our_rank], [our_score], s=70, color=BLUE, zorder=4)
    ax.annotate("our best construction\n26.70 → ~rank 40 of 4,252\n(had it been selected)",
                (our_rank, our_score), xytext=(our_rank - 20, our_score + 8),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("private-leaderboard rank (top 49 of 4,252)")
    ax.set_ylabel("private score")
    ax.set_ylim(24, 48)
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Where our best result would have landed", loc="left",
                  fontsize=13.5, color=INK, fontweight="bold", pad=12)
    save(fig, "fig11_rank_curve.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7(); fig8(); fig9(); fig10(); fig11()
    print("done ->", OUT)
