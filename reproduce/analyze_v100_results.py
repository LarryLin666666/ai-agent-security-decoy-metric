"""Turn cv_real.py's toktrue_results.csv / wallclock_results.csv into Working-Note-style
PNG figures (fig12/fig13), using the same palette as make_figures.py.
Run AFTER the V100 experiments: python analyze_v100_results.py [csv_dir] [out_dir]
Missing CSVs are skipped with a message, not an error, so this is safe to run partially.
"""
import csv
import os
import sys
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (same as make_figures.py) ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; BLUE = "#2a78d6"; ORANGE = "#eb6834"
AQUA = "#1baf7a"; RED = "#e34948"; GOOD = "#0ca30c"; CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 12,
})


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def _read_csv(path):
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"skip (empty): {path}")
        return None
    return rows


def fig_toktrue(csv_dir, out_dir):
    rows = _read_csv(os.path.join(csv_dir, "toktrue_results.csv"))
    if rows is None:
        return
    # mean true_gen_tokens per (model, variant), fired-only (unfired candidates are
    # compliance failures, not comparable token counts) - also track total N and fired
    # count per group so a total compliance failure can be labeled, not left blank.
    groups = {}
    totals = {}
    for r in rows:
        key = (r["model"], r["variant"])
        totals.setdefault(key, [0, 0])
        totals[key][1] += 1
        if r["fired"] != "1" or not r["true_gen_tokens"]:
            continue
        totals[key][0] += 1
        groups.setdefault(key, []).append(float(r["true_gen_tokens"]))
    if not groups:
        print("toktrue: no fired rows with a token count yet - nothing to plot")
        return

    variants = ["exfil_base", "exfil_shave", "conf_base", "conf_shave"]
    models = ["gpt_oss", "gemma"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(variants))
    w = 0.36
    for mi, (m, color) in enumerate(zip(models, [BLUE, AQUA])):
        vals = [stats.mean(groups.get((m, v), [float("nan")])) if (m, v) in groups else np.nan
                for v in variants]
        offs = x + (mi - 0.5) * w
        bars = ax.bar(offs, vals, width=w, color=color, zorder=3, label=m)
        for xi, v, vr in zip(offs, vals, variants):
            fired_n, total_n = totals.get((m, vr), [0, 0])
            if not np.isnan(v):
                ax.text(xi, v + 0.3, f"{v:.0f}", ha="center", fontsize=9.5, color=INK)
                if fired_n < total_n:
                    ax.text(xi, v + 2.2, f"({fired_n}/{total_n} fired)", ha="center",
                             fontsize=7.5, color=MUTED)
            elif total_n:
                ax.text(xi, 1.0, f"0/{total_n}\nfired", ha="center", va="bottom",
                         fontsize=8.5, color=CRITICAL, fontweight="bold")
    ax.set_xticks(x, ["EXFIL\nbase", "EXFIL\nshave", "CONFUSED\nbase", "CONFUSED\nshave"])
    ax.set_ylabel("true generated tokens / fired candidate (real tokenizer)")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("Does the writeup #15 token-shave actually reduce our token count?",
                  loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    path = os.path.join(out_dir, "fig12_toktrue.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def fig_wallclock(csv_dir, out_dir):
    rows = _read_csv(os.path.join(csv_dir, "wallclock_results.csv"))
    if rows is None:
        return
    groups = {}
    for r in rows:
        key = (r["model"], r["construction"], r["guardrail"])
        groups.setdefault(key, []).append(float(r["wall_s"]))
    models = sorted({r["model"] for r in rows})
    guards = ["allow", "optimal", "rules"]
    for m in models:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(guards))
        w = 0.36
        max_val = 0.0
        for ci, (cname, color) in enumerate(zip(["exfil", "confused"], [RED, GOOD])):
            vals = [stats.mean(groups.get((m, cname, g), [float("nan")])) for g in guards]
            max_val = max([max_val] + [v for v in vals if not np.isnan(v)])
            offs = x + (ci - 0.5) * w
            ax.bar(offs, vals, width=w, color=color, zorder=3, label=cname)
            for xi, v in zip(offs, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.05, f"{v:.2f}s", ha="center", fontsize=9.5, color=INK)
        ax.set_xticks(x, ["allow-all", "OptimalGuardrail\n(public)", "RulesGuardrail\n(denies marker)"])
        ax.set_ylabel("wall-clock / candidate (s)")
        _clean(ax)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_ylim(0, max_val * 1.32 if max_val else 1)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
        ax.set_title(f"{m}: does a hop-0 denial actually shorten wall-clock?",
                      loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=12)
        path = os.path.join(out_dir, f"fig13_wallclock_{m}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print("wrote", path)


def fig_injection_matrix_realn(csv_dir, out_dir):
    """Same 2x4 model x encoding grid as Fig 5, but with real fire-RATE percentages
    from a large-N encinject_results.csv rerun instead of the small illustrative N."""
    rows = _read_csv(os.path.join(csv_dir, "encinject_results.csv"))
    if rows is None:
        return
    cols_map = {"plain_e1": "plain text", "morse_e6": "Morse", "base64_e7": "base64", "rot13_e8": "ROT13"}
    cols = list(cols_map.keys())
    model_rows = {"gpt_oss": 0, "gemma": 1}
    grid_rate = np.zeros((2, 4))
    grid_n = np.zeros((2, 4))
    action_preds = ("EXFILTRATION", "UNTRUSTED_TO_ACTION", "DESTRUCTIVE_WRITE")
    for r in rows:
        if r["model"] not in model_rows or r["encoding"] not in cols_map:
            continue
        i, j = model_rows[r["model"]], cols.index(r["encoding"])
        grid_n[i, j] += 1
        preds = r["predicates"] or ""
        if any(p in preds for p in action_preds):
            grid_rate[i, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(grid_n > 0, 100 * grid_rate / np.maximum(grid_n, 1), np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for i in range(2):
        for j in range(4):
            p = pct[i, j]
            fires = (not np.isnan(p)) and p > 0
            color = CRITICAL if fires else "#eceae4"
            txt_color = "white" if fires else MUTED
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.92, 0.92, facecolor=color,
                                        edgecolor=SURFACE, linewidth=3))
            label = f"{p:.0f}% fire" if not np.isnan(p) else "no data"
            ax.text(j + 0.46, 1 - i + 0.46, label, ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="bold")
    ax.set_xlim(0, 4); ax.set_ylim(0, 2.2)
    ax.set_xticks([0.46, 1.46, 2.46, 3.46], list(cols_map.values()), fontsize=11, color=INK2)
    ax.set_yticks([1.46, 0.46], ["gpt-oss", "Gemma 4"], fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    n_each = int(np.nanmax(grid_n)) if grid_n.size else 0
    ax.set_title(f"Acts on the injected instruction? N={n_each}/cell (large-N rerun)", loc="left",
                 fontsize=12.5, color=INK, fontweight="bold", pad=12)
    path = os.path.join(out_dir, "fig14_injection_matrix_realn.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def fig_untrusted_realn(csv_dir, out_dir):
    """email_1/2/3 + websearch, under allow-all and OptimalGuardrail, N=60 rerun."""
    rows = _read_csv(os.path.join(csv_dir, "untrusted_results.csv"))
    if rows is None:
        return
    srcs = ["email_1", "email_2", "email_3", "websearch"]
    action_preds = ("EXFILTRATION", "UNTRUSTED_TO_ACTION", "DESTRUCTIVE_WRITE")
    grid_rate = {}
    grid_n = {}
    for r in rows:
        key = (r["model"], r["guardrail"], r["source"])
        grid_n[key] = grid_n.get(key, 0) + 1
        preds = r["predicates"] or ""
        if any(p in preds for p in action_preds):
            grid_rate[key] = grid_rate.get(key, 0) + 1
    models = sorted({r["model"] for r in rows})
    for m in models:
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        for gi, g in enumerate(["allow", "optimal"]):
            for si, s in enumerate(srcs):
                n = grid_n.get((m, g, s), 0)
                fired = grid_rate.get((m, g, s), 0)
                pct = 100 * fired / n if n else float("nan")
                fires = (not np.isnan(pct)) and pct > 0
                color = CRITICAL if fires else "#eceae4"
                txt_color = "white" if fires else MUTED
                x = si
                y = 1 - gi
                ax.add_patch(plt.Rectangle((x, y), 0.92, 0.92, facecolor=color,
                                            edgecolor=SURFACE, linewidth=3))
                label = f"{pct:.0f}% fire" if n else "no data"
                ax.text(x + 0.46, y + 0.46, label, ha="center", va="center",
                        fontsize=10.5, color=txt_color, fontweight="bold")
        ax.set_xlim(0, 4); ax.set_ylim(0, 2.2)
        ax.set_xticks([0.46, 1.46, 2.46, 3.46], ["email_1", "email_2", "email_3", "websearch"],
                      fontsize=10.5, color=INK2)
        ax.set_yticks([1.46, 0.46], ["allow-all", "OptimalGuardrail\n(public)"], fontsize=10, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{m}: acts on the injected source? N=60/cell", loc="left",
                     fontsize=12.5, color=INK, fontweight="bold", pad=12)
        path = os.path.join(out_dir, f"fig15_untrusted_realn_{m}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print("wrote", path)


def fig_mhchain(csv_dir, out_dir):
    """New never-before-tested mh_ multi-hop fixture chains: does routing the last hop
    through a fresh web.open() (instead of an intervening fs.read) finally reach
    UNTRUSTED_TO_ACTION reliably? model x guardrail x source grid."""
    rows = _read_csv(os.path.join(csv_dir, "mhchain_results.csv"))
    if rows is None:
        return
    srcs = ["mh_email_1", "mh_email_lure_2"]
    action_preds = ("EXFILTRATION", "UNTRUSTED_TO_ACTION", "DESTRUCTIVE_WRITE")
    grid_rate, grid_n = {}, {}
    for r in rows:
        key = (r["model"], r["guardrail"], r["source"])
        grid_n[key] = grid_n.get(key, 0) + 1
        preds = r["predicates"] or ""
        if any(p in preds for p in action_preds):
            grid_rate[key] = grid_rate.get(key, 0) + 1
    models = sorted({r["model"] for r in rows})
    for m in models:
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        for gi, g in enumerate(["allow", "optimal"]):
            for si, s in enumerate(srcs):
                n = grid_n.get((m, g, s), 0)
                fired = grid_rate.get((m, g, s), 0)
                pct = 100 * fired / n if n else float("nan")
                fires = (not np.isnan(pct)) and pct > 0
                color = CRITICAL if fires else "#eceae4"
                txt_color = "white" if fires else MUTED
                x, y = si, 1 - gi
                ax.add_patch(plt.Rectangle((x, y), 0.92, 0.92, facecolor=color,
                                            edgecolor=SURFACE, linewidth=3))
                label = f"{pct:.0f}% fire" if n else "no data"
                ax.text(x + 0.46, y + 0.46, label, ha="center", va="center",
                        fontsize=11, color=txt_color, fontweight="bold")
        ax.set_xlim(0, 2); ax.set_ylim(0, 2.2)
        ax.set_xticks([0.46, 1.46], ["mh_email_1\n(file+web hop)", "mh_email_lure_2\n(web+web hop)"],
                      fontsize=10, color=INK2)
        ax.set_yticks([1.46, 0.46], ["allow-all", "OptimalGuardrail\n(public)"], fontsize=10, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{m}: multi-hop chain reaches UNTRUSTED_TO_ACTION? N=20/cell", loc="left",
                     fontsize=12, color=INK, fontweight="bold", pad=12)
        path = os.path.join(out_dir, f"fig16_mhchain_{m}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print("wrote", path)


def fig4_verify(csv_dir, out_dir):
    """Re-verify Fig 4 (destructive-write K-stacking) at N=20 instead of the original
    small N. Same chart shape as make_figures.fig4."""
    rows = _read_csv(os.path.join(csv_dir, "destmulti_results.csv"))
    if rows is None:
        return
    ks = [1, 2, 3, 5, 6, 7]
    by_mk = {}
    for r in rows:
        key = (r["model"], int(r["k"]))
        by_mk.setdefault(key, []).append(float(r["destructive_count"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(ks, ks, color=AXIS, lw=1.4, ls=(0, (4, 3)), zorder=2, label="ideal  y = K")
    for mname, color in [("gemma", AQUA), ("gpt_oss", BLUE)]:
        ys = [stats.mean(by_mk.get((mname, k), [float("nan")])) for k in ks]
        n0 = len(by_mk.get((mname, ks[0]), []))
        ax.plot(ks, ys, color=color, lw=2.2, marker="o", ms=6, zorder=3, label=f"{mname}  (N={n0}/K)")
    ax.set_xticks(ks, [f"K={k}" for k in ks])
    ax.set_ylabel("writes credited / candidate (mean)")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Fig 4 re-verified at N=20 (was N=6): does the stacking hold?", loc="left",
                 fontsize=12.5, color=INK, fontweight="bold", pad=12)
    path = os.path.join(out_dir, "fig17_destmulti_verify.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def fig7_verify(csv_dir, out_dir):
    """Re-verify Fig 7 (read-and-post throughput vs K) at N=20 instead of the original N=5."""
    rows = _read_csv(os.path.join(csv_dir, "readpostmp2msg_results.csv"))
    if rows is None:
        return
    ks = [1, 2, 4, 6]
    by_mk = {}
    n_by_mk = {}
    for r in rows:
        key = (r["model"], int(r["k"]))
        exfil = float(r["exfil_count"])
        turns = float(r["turns"])
        by_mk.setdefault(key, []).append((exfil, turns))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for mname, color in [("gemma", AQUA), ("gpt_oss", BLUE)]:
        ys = []
        for k in ks:
            pts = by_mk.get((mname, k), [])
            if not pts:
                ys.append(float("nan")); continue
            exfil_sum = sum(p[0] for p in pts); turns_sum = sum(p[1] for p in pts)
            raw = (exfil_sum / len(pts)) * 16 + 2
            gpc = turns_sum / len(pts)
            ys.append(raw / max(1e-6, gpc))
        n0 = len(by_mk.get((mname, ks[0]), []))
        ax.plot(ks, ys, color=color, lw=2.2, marker="o", ms=6, zorder=3, label=f"{mname}  (N={n0}/K)")
        for xi, yi in zip(ks, ys):
            if not np.isnan(yi):
                ax.text(xi, yi + 0.2, f"{yi:.1f}", ha="center", fontsize=9, color=color)
    ax.set_xticks(ks, [f"K={k}" for k in ks])
    ax.set_ylabel("raw / generation")
    _clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Fig 7 re-verified at N=20 (was N=5): does the throughput hold?", loc="left",
                 fontsize=12.5, color=INK, fontweight="bold", pad=12)
    path = os.path.join(out_dir, "fig18_readpost_verify.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def fig_cdshave(csv_dir, out_dir):
    """fig19: CONFUSED_DEPUTY token-shave vs Kohei/#4 tricks. Grouped bars per model:
    avg true generated tokens over FIRING candidates (a shave that broke firing is not a
    shave), annotated with fire-rate. Compares conf_base vs the shave variants; the
    'tension' variant (direct header forge) is shown by fire-rate only."""
    rows = _read_csv(os.path.join(csv_dir, "cdshave_results.csv"))
    if rows is None:
        return
    # aggregate
    agg = {}   # (model,variant) -> dict(kind, n, fired, toks[list over firing])
    order = {}
    for r in rows:
        key = (r["model"], r["variant"])
        d = agg.setdefault(key, {"kind": r.get("kind", "shave"), "n": 0, "fired": 0, "toks": []})
        d["kind"] = r.get("kind", d["kind"])
        d["n"] += 1
        fired = str(r.get("fired", "")) in ("1", "1.0", "True")
        d["fired"] += int(fired)
        tv = r.get("true_gen_tokens", "")
        if fired and tv not in ("", None):
            try: d["toks"].append(float(tv))
            except ValueError: pass
        order.setdefault(r["model"], []).append(r["variant"])
    models = list(order.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.4), squeeze=False)
    for ax, mname in zip(axes[0], models):
        variants = list(dict.fromkeys(order[mname]))
        xs = np.arange(len(variants))
        vals, labels, cols = [], [], []
        for v in variants:
            d = agg[(mname, v)]
            avg = stats.mean(d["toks"]) if d["toks"] else 0.0
            vals.append(avg)
            fr = d["fired"] / d["n"] if d["n"] else 0
            labels.append(f"{avg:.1f} tok\n{fr:.0%} fire")
            cols.append(BLUE if v == "conf_base" else (AQUA if d["kind"] == "shave" else MUTED))
        ax.bar(xs, vals, width=0.6, color=cols, zorder=3)
        for xi, v, lab in zip(xs, vals, labels):
            ax.text(xi, v + 0.4, lab, ha="center", fontsize=9, color=INK)
        ax.set_xticks(xs, variants, fontsize=9, rotation=12)
        ax.set_ylabel("avg true gen-tokens (firing cands)")
        _clean(ax); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(mname, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    fig.suptitle("Fig 19 — CONFUSED_DEPUTY token-shave: do #4's Gemma ASCII-quote / gpt-oss tricks reproduce?",
                 x=0.02, ha="left", fontsize=12.6, fontweight="bold", color=INK)
    path = os.path.join(out_dir, "fig19_cdshave.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25); plt.close(fig)
    print("wrote", path)


def fig_wrapupcollapse(csv_dir, out_dir):
    """fig20: gpt-oss wrap-up-collapse terminal (Liauuuuu's soft in-forge terminal).
    Per predicate arm (EXFIL, CONFUSED_DEPUTY): avg true gen-tokens + fire-rate for
    empty-forge baseline vs in-forge terminal vs too-strict terminal. Tests whether the
    soft terminal cuts tokens at unchanged fire, and whether the strict one hits the
    'protocol floor' (fire -> 0). wall_s is intentionally NOT plotted (non-transferable)."""
    rows = _read_csv(os.path.join(csv_dir, "wrapupcollapse_results.csv"))
    if rows is None:
        return
    agg = {}
    for r in rows:
        key = (r["variant"],)
        d = agg.setdefault(key, {"n": 0, "fired": 0, "toks": []})
        d["n"] += 1
        fired = str(r.get("fired", "")) in ("1", "1.0", "True")
        d["fired"] += int(fired)
        tv = r.get("true_gen_tokens", "")
        if fired and tv not in ("", None):
            try: d["toks"].append(float(tv))
            except ValueError: pass
    arms = [("EXFILTRATION", ["exfil_empty_baseline", "exfil_term_inforge", "exfil_term_strict"]),
            ("CONFUSED_DEPUTY", ["conf_empty_baseline", "conf_term_inforge", "conf_term_strict"])]
    short = {"exfil_empty_baseline": "empty\n(baseline)", "exfil_term_inforge": "in-forge\nterminal",
             "exfil_term_strict": "strict\nterminal", "conf_empty_baseline": "empty\n(baseline)",
             "conf_term_inforge": "in-forge\nterminal", "conf_term_strict": "strict\nterminal"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), squeeze=False)
    for ax, (pred, variants) in zip(axes[0], arms):
        variants = [v for v in variants if (v,) in agg]
        xs = np.arange(len(variants))
        vals, cols, labs = [], [], []
        for v in variants:
            d = agg[(v,)]
            avg = stats.mean(d["toks"]) if d["toks"] else 0.0
            fr = d["fired"] / d["n"] if d["n"] else 0
            vals.append(avg)
            cols.append(BLUE if "empty" in v else (AQUA if "inforge" in v else CRITICAL))
            labs.append(f"{avg:.1f} tok\n{fr:.0%} fire")
        ax.bar(xs, vals, width=0.6, color=cols, zorder=3)
        for xi, v, lab in zip(xs, vals, labs):
            ax.text(xi, v + 0.4, lab, ha="center", fontsize=9, color=INK)
        ax.set_xticks(xs, [short.get(v, v) for v in variants], fontsize=9)
        ax.set_ylabel("avg true gen-tokens (firing cands)")
        _clean(ax); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(pred, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    fig.suptitle("Fig 20 — Soft in-forge wrap-up terminal (gpt-oss): does it cut tokens without breaking firing?",
                 x=0.02, ha="left", fontsize=12.6, fontweight="bold", color=INK)
    path = os.path.join(out_dir, "fig20_wrapupcollapse.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25); plt.close(fig)
    print("wrote", path)


def fig_eogmargin(csv_dir, out_dir):
    """fig21: our own read of the post-tool (hop-1) first-token EOG rank via top_logprobs.
    Bars = fraction of candidates whose end-of-generation token is inside the top-K at the
    first post-tool position, per model (averaged over exfil+conf). gpt-oss's EOG sits far
    outside the top-K (consistent with xz -38 / #5 rank 128); Gemma's is near the top
    (xz -14 / #5 rank 2). Annotated with mean rank/gap when in top-K."""
    rows = _read_csv(os.path.join(csv_dir, "eogmargin_results.csv"))
    if rows is None:
        return
    from collections import defaultdict
    topk = os.environ.get("EOG_TOPK", "20")
    agg = defaultdict(lambda: {"n": 0, "in": 0, "ranks": [], "gaps": []})
    for r in rows:
        d = agg[r["model"]]
        d["n"] += 1
        if str(r.get("eog_in_topk", "")) == "1":
            d["in"] += 1
            try: d["ranks"].append(float(r["eog_rank"]))
            except (ValueError, TypeError): pass
            try: d["gaps"].append(float(r["gap"]))
            except (ValueError, TypeError): pass
    models = [m for m in ["gpt_oss", "gemma"] if m in agg]
    if not models:
        print("skip (no eogmargin models)"); return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(models))
    rates = [agg[m]["in"] / agg[m]["n"] * 100 if agg[m]["n"] else 0 for m in models]
    cols = [BLUE if m == "gpt_oss" else AQUA for m in models]
    ax.bar(xs, rates, width=0.5, color=cols, zorder=3)
    for xi, m in zip(xs, models):
        d = agg[m]
        rate = d["in"] / d["n"] * 100 if d["n"] else 0
        mr = stats.mean(d["ranks"]) if d["ranks"] else None
        mg = stats.mean(d["gaps"]) if d["gaps"] else None
        lab = f"{rate:.0f}% in top-{topk}"
        if mr is not None: lab += f"\nmean rank {mr:.1f}"
        if mg is not None: lab += f"\nmean gap {mg:.1f}"
        ax.text(xi, rate + 2, lab, ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs, models, fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_ylabel(f"post-tool EOG token in top-{topk} (%)")
    _clean(ax); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Fig 21 — Post-tool EOG margin (our read): gpt-oss's wrap-up EOG is far from greedy, Gemma's is near",
                 loc="left", fontsize=11.6, color=INK, fontweight="bold", pad=12)
    path = os.path.join(out_dir, "fig21_eogmargin.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    fig_eogmargin(csv_dir, out_dir)
    fig_toktrue(csv_dir, out_dir)
    fig_wallclock(csv_dir, out_dir)
    fig_injection_matrix_realn(csv_dir, out_dir)
    fig_untrusted_realn(csv_dir, out_dir)
    fig_mhchain(csv_dir, out_dir)
    fig4_verify(csv_dir, out_dir)
    fig7_verify(csv_dir, out_dir)
    fig_cdshave(csv_dir, out_dir)
    fig_wrapupcollapse(csv_dir, out_dir)
