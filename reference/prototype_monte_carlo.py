"""
Probabilistic DDInSAR fringe-count analysis under acquisition-time jitter.

For each of the 6 nominal scenarios, perturb the three acquisition times
independently by dt ~ U(-1 h, +1 h) and recompute:
    h_DD = h(t1+d1) - 2 h(t2+d2) + h(t3+d3)
    N_fringe = |h_DD| * cos(theta) / (lambda/2)
Monte Carlo with N=10,000 realisations per scenario -> distribution of
fringe counts, stability metrics, and probability of falling below
mapping-useful thresholds.

Output:
  - ddinsar_monte_carlo.png / .svg (panel of histograms + summary)
  - ddinsar_monte_carlo_stats.csv (full summary table)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ---- constants (match prior simulation) ----
lam_L   = 0.2360           # L-band wavelength (m)
half_lam = lam_L / 2.0     # 11.8 cm -- one LOS fringe
theta_i = np.deg2rad(39.0)
cos_i   = np.cos(theta_i)

def tide(t_hours, A_M2=0.80, A_K1=0.50, phi_M2=0.0, phi_K1=1.1):
    T_M2 = 12.42; T_K1 = 23.93
    return (A_M2 * np.cos(2*np.pi*t_hours/T_M2 + phi_M2)
          + A_K1 * np.cos(2*np.pi*t_hours/T_K1 + phi_K1))

# six nominal scenarios, renamed by operational meaning
# (ordered from worst to best triplet for GL mapping)
scenarios = [
    ("NULL",     212.0),   # near-zero h_DD, GL invisible
    ("WEAK",     200.0),   # ~2 fringes, marginal
    ("MARGINAL",   6.0),   # ~3 fringes, threshold-quality
    ("GOOD",     142.0),   # ~5 fringes, reliable mapping
    ("STRONG",   166.0),   # ~6 fringes, clean
    ("PEAK",     128.0),   # maximum h_DD, densest fringes
]

# ---- Monte Carlo ----
N = 10000
rng = np.random.default_rng(2026)

results = {}
for name, t0 in scenarios:
    # three independent timing jitters in HOURS, uniform in [-1, +1]
    d1 = rng.uniform(-1, 1, N)
    d2 = rng.uniform(-1, 1, N)
    d3 = rng.uniform(-1, 1, N)

    h1 = tide(t0      + d1)
    h2 = tide(t0 + 12*24 + d2)
    h3 = tide(t0 + 24*24 + d3)
    hDD = h1 - 2*h2 + h3
    nfr = np.abs(hDD) * cos_i / half_lam   # |fringes|

    # nominal (zero jitter)
    h1_0 = tide(t0); h2_0 = tide(t0+12*24); h3_0 = tide(t0+24*24)
    hDD_0 = h1_0 - 2*h2_0 + h3_0
    nfr_0 = abs(hDD_0) * cos_i / half_lam

    results[name] = {
        "t0": t0,
        "hDD_nominal": hDD_0,
        "nfr_nominal": nfr_0,
        "hDD_samples": hDD,
        "nfr_samples": nfr,
    }

# ---- Statistics per scenario ----
stats_rows = []
for name, r in results.items():
    s = r["nfr_samples"]
    hDD = r["hDD_samples"]
    row = {
        "scenario": name,
        "t0_h": r["t0"],
        "hDD_nominal_m": r["hDD_nominal"],
        "nfr_nominal": r["nfr_nominal"],
        "nfr_mean": float(np.mean(s)),
        "nfr_median": float(np.median(s)),
        "nfr_std": float(np.std(s)),
        "nfr_p05": float(np.percentile(s, 5)),
        "nfr_p95": float(np.percentile(s, 95)),
        "nfr_min": float(np.min(s)),
        "nfr_max": float(np.max(s)),
        # operational thresholds
        "P_below_1_fringe": float(np.mean(s < 1.0)),
        "P_below_2_fringes": float(np.mean(s < 2.0)),
        "P_above_3_fringes": float(np.mean(s >= 3.0)),
        # sign flip probability (does DD residual change sign?)
        "P_sign_flip": float(np.mean(np.sign(hDD) != np.sign(r["hDD_nominal"]))
                              if r["hDD_nominal"] != 0 else np.nan),
    }
    stats_rows.append(row)

# ---- Write CSV ----
csv_path = "/home/claude/ddinsar_monte_carlo_stats.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
    w.writeheader()
    for row in stats_rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in row.items()})
print(f"Wrote {csv_path}")

# ---- Print summary ----
print("\n" + "="*110)
print(f"Monte Carlo over acquisition-time jitter  (dt ~ U[-1,+1] h, N={N})")
print("="*110)
hdr = f"{'scenario':>10} {'t0(h)':>6} {'nom hDD':>8} {'nom nfr':>8} {'mean':>6} {'med':>6} {'std':>6} " \
      f"{'p5':>6} {'p95':>6} {'P<1fr':>7} {'P<2fr':>7} {'P>=3fr':>7} {'P flip':>7}"
print(hdr); print("-"*118)
for row in stats_rows:
    print(f"{row['scenario']:>10} {row['t0_h']:6.0f} "
          f"{row['hDD_nominal_m']:+8.3f} {row['nfr_nominal']:8.2f} "
          f"{row['nfr_mean']:6.2f} {row['nfr_median']:6.2f} {row['nfr_std']:6.2f} "
          f"{row['nfr_p05']:6.2f} {row['nfr_p95']:6.2f} "
          f"{row['P_below_1_fringe']:7.3f} {row['P_below_2_fringes']:7.3f} "
          f"{row['P_above_3_fringes']:7.3f} {row['P_sign_flip']:7.3f}")

# ========== Plot ==========
fig = plt.figure(figsize=(15, 11), facecolor="white")
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32,
                      height_ratios=[1, 1, 1.05])

colors = plt.get_cmap("tab10").colors

# --- 6 histograms (top 2 rows) ---
for i, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[i//3, i%3])
    s = r["nfr_samples"]
    nom = r["nfr_nominal"]

    # histogram
    n_bins = 50
    ax.hist(s, bins=n_bins, color=colors[i], alpha=0.75,
            edgecolor="white", linewidth=0.5, density=True)

    # nominal value
    ax.axvline(nom, color="k", lw=2, label=f"nominal = {nom:.2f}")
    ax.axvline(np.mean(s), color="k", lw=1.2, ls="--",
               label=f"mean = {np.mean(s):.2f}")
    ax.axvspan(np.percentile(s, 5), np.percentile(s, 95),
               color="k", alpha=0.10, label="5–95 %")

    # operational thresholds
    ax.axvline(1.0, color="#c72b2b", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(3.0, color="#2e8b57", lw=0.8, ls=":", alpha=0.6)

    # coloured border
    for sp in ax.spines.values():
        sp.set_edgecolor(colors[i]); sp.set_linewidth(2.2)

    ax.set_title(f"{name}  (t$_0$ = {r['t0']:.0f} h)\n"
                 f"h$_{{DD}}$ nominal = {r['hDD_nominal']:+.2f} m",
                 fontsize=10.5)
    ax.set_xlabel("|fringes|  (LOS)", fontsize=9)
    ax.set_ylabel("prob. density", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

# --- Bottom: side-by-side comparison (boxplot + summary table) ---
ax_box = fig.add_subplot(gs[2, :2])
data = [r["nfr_samples"] for _, r in results.items()]
bp = ax_box.boxplot(data, labels=[n for n, _ in scenarios],
                    showmeans=True, meanline=True,
                    patch_artist=True, widths=0.55,
                    whis=(5, 95), showfliers=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("k"); median.set_linewidth(1.5)
for mean in bp["means"]:
    mean.set_color("k"); mean.set_linestyle("--")

# overlay nominal values
for i, (name, r) in enumerate(results.items()):
    ax_box.plot(i+1, r["nfr_nominal"], marker="D", color="k",
                ms=8, mec="w", mew=1.2, zorder=10,
                label="nominal" if i == 0 else None)

ax_box.axhline(1.0, color="#c72b2b", lw=0.8, ls=":",
               label="1 fringe (detectability floor)")
ax_box.axhline(3.0, color="#2e8b57", lw=0.8, ls=":",
               label="3 fringes (good GL mapping)")
ax_box.set_ylabel("|fringes|  (LOS)")
ax_box.set_xlabel("Scenario")
ax_box.set_title("Fringe-count distributions under ±1 h acquisition-time jitter\n"
                 f"(Monte Carlo, N={N} per scenario;  box = IQR,  whiskers = 5–95 %)",
                 fontsize=11)
ax_box.legend(fontsize=8.5, loc="upper left")
ax_box.grid(alpha=0.3, axis="y")
plt.setp(ax_box.get_xticklabels(), rotation=15, ha="right", fontsize=10)

# --- Summary text panel ---
ax_tab = fig.add_subplot(gs[2, 2])
ax_tab.axis("off")
txt = "Stability summary\n" + "─"*32 + "\n"
txt += f"{'':>9} {'mean':>5} {'σ':>5} {'P<1fr':>7}\n"
for row in stats_rows:
    flag = " ⚠" if row["P_below_1_fringe"] > 0.3 else ""
    txt += (f"{row['scenario']:>9} {row['nfr_mean']:>5.2f} "
            f"{row['nfr_std']:>5.2f} {row['P_below_1_fringe']:>7.3f}{flag}\n")
txt += "\n⚠  = unreliable for GL mapping\n     (>30% chance of <1 fringe)"
ax_tab.text(0.02, 0.98, txt, transform=ax_tab.transAxes,
            family="monospace", fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f5f5f5", ec="#bbb"))

fig.suptitle("Probabilistic DDInSAR fringe-count analysis  —  L-band, 12-day repeat, "
             "±1 h acquisition-time jitter per epoch",
             fontsize=13, y=0.995)

out_svg = "/home/claude/ddinsar_monte_carlo.svg"
out_png = "/home/claude/ddinsar_monte_carlo.png"
plt.savefig(out_svg, format="svg", bbox_inches="tight")
plt.savefig(out_png, format="png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_svg}")
print(f"Wrote {out_png}")
