"""
Rigid triplet sweep: shift the ENTIRE 3-date triplet by Delta_t together,
keeping t2-t1 = t3-t2 = 12 days fixed. Step Delta_t by 1 hour over
a full tidal envelope (we use the lunar month, 29.53 days, to capture
both M2 and K1 phase relationships).

This is NOT an uncertainty analysis; it is a *mission-planning*
sensitivity: for every possible triplet phase, what is the resulting
DD fringe count?
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ---- constants ----
lam_L   = 0.2360
half_lam = lam_L / 2.0
theta_i = np.deg2rad(39.0)
cos_i   = np.cos(theta_i)

def tide(t_hours, A_M2=0.80, A_K1=0.50, phi_M2=0.0, phi_K1=1.1):
    T_M2 = 12.42; T_K1 = 23.93
    return (A_M2 * np.cos(2*np.pi*t_hours/T_M2 + phi_M2)
          + A_K1 * np.cos(2*np.pi*t_hours/T_K1 + phi_K1))

# ---- Sweep ----
# Step 1 h over 29.53 days = ~709 steps. Triplet spacing fixed at 12 d.
dt_step = 1.0   # hours
T_sweep_h = 29.53 * 24  # full lunar month
Delta_t = np.arange(0, T_sweep_h + dt_step, dt_step)  # 0..709 h

h1 = tide(Delta_t)
h2 = tide(Delta_t + 12*24)
h3 = tide(Delta_t + 24*24)
hDD = h1 - 2*h2 + h3
nfr = np.abs(hDD) * cos_i / half_lam

# ---- Statistics over the sweep ----
stats = {
    "N_triplets": len(Delta_t),
    "sweep_hours": T_sweep_h,
    "hDD_mean": float(np.mean(hDD)),
    "hDD_std":  float(np.std(hDD)),
    "hDD_min":  float(np.min(hDD)),
    "hDD_max":  float(np.max(hDD)),
    "nfr_mean":   float(np.mean(nfr)),
    "nfr_median": float(np.median(nfr)),
    "nfr_std":    float(np.std(nfr)),
    "nfr_p05":    float(np.percentile(nfr, 5)),
    "nfr_p50":    float(np.percentile(nfr, 50)),
    "nfr_p95":    float(np.percentile(nfr, 95)),
    "nfr_max":    float(np.max(nfr)),
    # operational probabilities (fraction of sweep time)
    "frac_below_1fr":  float(np.mean(nfr < 1.0)),
    "frac_below_2fr":  float(np.mean(nfr < 2.0)),
    "frac_below_3fr":  float(np.mean(nfr < 3.0)),
    "frac_above_3fr":  float(np.mean(nfr >= 3.0)),
    "frac_above_5fr":  float(np.mean(nfr >= 5.0)),
    "frac_null_zone":  float(np.mean(nfr < 0.5)),   # truly unusable
}

# Classify each triplet in the sweep
def classify(n):
    if n < 0.5:  return "NULL"
    if n < 2.0:  return "WEAK"
    if n < 3.0:  return "MARGINAL"
    if n < 5.0:  return "GOOD"
    if n < 7.0:  return "STRONG"
    return "PEAK"

classes = np.array([classify(n) for n in nfr])
unique, counts = np.unique(classes, return_counts=True)
class_fractions = dict(zip(unique, counts / len(classes)))

print("="*70)
print(f"Rigid triplet sweep: Delta_t from 0 to {T_sweep_h:.1f} h, step {dt_step} h")
print(f"N triplets examined: {len(Delta_t)}")
print("="*70)
print(f"h_DD range:    [{stats['hDD_min']:+.3f}, {stats['hDD_max']:+.3f}] m")
print(f"h_DD std:       {stats['hDD_std']:.3f} m")
print(f"Fringes: mean={stats['nfr_mean']:.2f}  median={stats['nfr_median']:.2f}  "
      f"std={stats['nfr_std']:.2f}  max={stats['nfr_max']:.2f}")
print(f"\nFraction of time-of-month in each class:")
for cls in ["NULL","WEAK","MARGINAL","GOOD","STRONG","PEAK"]:
    frac = class_fractions.get(cls, 0.0)
    bar = "█" * int(round(frac*40))
    print(f"  {cls:>9}: {frac*100:5.1f}%  {bar}")
print(f"\nOperational probabilities:")
print(f"  P(<1 fringe)  = {stats['frac_below_1fr']*100:5.1f}%   'unusable'")
print(f"  P(<2 fringes) = {stats['frac_below_2fr']*100:5.1f}%")
print(f"  P(<3 fringes) = {stats['frac_below_3fr']*100:5.1f}%")
print(f"  P(>=3 fringes)= {stats['frac_above_3fr']*100:5.1f}%   'good mapping'")
print(f"  P(>=5 fringes)= {stats['frac_above_5fr']*100:5.1f}%   'robust'")

# ---- Find the specific Delta_t values of each earlier-named scenario, for reference ----
named = {"NULL":212.0, "WEAK":200.0, "MARGINAL":6.0, "GOOD":142.0,
         "STRONG":166.0, "PEAK":128.0}

# ---- CSV output ----
csv_path = "/home/claude/ddinsar_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["delta_t_h","h1_m","h2_m","h3_m","h_DD_m","fringes","class"])
    for dt, a, b, c, hd, n, cl in zip(Delta_t, h1, h2, h3, hDD, nfr, classes):
        w.writerow([f"{dt:.1f}", f"{a:+.4f}", f"{b:+.4f}", f"{c:+.4f}",
                    f"{hd:+.4f}", f"{n:.3f}", cl])
print(f"\nWrote per-step CSV: {csv_path}")

# ================ Plot ================
fig = plt.figure(figsize=(15, 11), facecolor="white")
gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.30,
                      height_ratios=[0.9, 1.1, 1.0, 1.0])

# Row 1: tide time series with ALL triplet acquisition triplets shown as thin lines
ax1 = fig.add_subplot(gs[0, :])
t_plot = np.linspace(0, T_sweep_h + 48, 5000)
ax1.plot(t_plot/24, tide(t_plot), color="#1f4e79", lw=1.1, zorder=3)
ax1.axhline(0, color="k", lw=0.4)
ax1.set_xlabel("Time since epoch (days)")
ax1.set_ylabel("Tide (m)")
ax1.set_title(f"Tide signal over one lunar month. Triplet is a rigid 3-epoch window "
              f"(Δt, Δt+12 d, Δt+24 d) swept by 1 h steps  ({len(Delta_t)} triplets)",
              fontsize=11)
# highlight one example triplet
for dt in [0, 100, 300, 500]:
    ys = [tide(dt), tide(dt+12*24), tide(dt+24*24)]
    xs = [dt/24, (dt+12*24)/24, (dt+24*24)/24]
    ax1.plot(xs, ys, "o-", color="#c72b2b", alpha=0.35, ms=5, lw=0.8)
ax1.set_xlim(-0.3, 26)
ax1.grid(alpha=0.3)

# Row 2: sweep curves — h_DD and fringes vs Delta_t
ax2 = fig.add_subplot(gs[1, :])
ax2b = ax2.twinx()

ax2.plot(Delta_t/24, hDD, color="#1f4e79", lw=1.3, label="h$_{DD}$")
ax2.fill_between(Delta_t/24, 0, hDD, color="#1f4e79", alpha=0.15)
ax2.axhline(0, color="k", lw=0.4)
ax2.set_xlabel("Triplet start offset Δt (days)")
ax2.set_ylabel("h$_{DD}$ (m)", color="#1f4e79")
ax2.tick_params(axis='y', labelcolor="#1f4e79")

ax2b.plot(Delta_t/24, nfr, color="#c72b2b", lw=1.1, alpha=0.85, label="|fringes|")
ax2b.set_ylabel("|fringes|  (1 fringe = 11.8 cm LOS)", color="#c72b2b")
ax2b.tick_params(axis='y', labelcolor="#c72b2b")
ax2b.axhline(1, color="#c72b2b", lw=0.6, ls=":", alpha=0.5)
ax2b.axhline(3, color="#2e8b57", lw=0.6, ls=":", alpha=0.5)

# mark the previously named scenarios on this curve
class_colors = {"NULL":"#4c78a8","WEAK":"#f58518","MARGINAL":"#54a24b",
                "GOOD":"#e45756","STRONG":"#b279a2","PEAK":"#8b6f47"}
for cl, dt_val in named.items():
    idx = int(round(dt_val/dt_step))
    ax2b.plot(dt_val/24, nfr[idx], "o", color=class_colors[cl], ms=9,
              mec="k", mew=0.8, zorder=10)
    ax2b.annotate(cl, (dt_val/24, nfr[idx]),
                  xytext=(4, 6), textcoords="offset points",
                  fontsize=8.5, fontweight="bold", color=class_colors[cl])

ax2.set_title("Rigid-triplet sensitivity sweep: fringe count vs triplet start time",
              fontsize=11)
ax2.set_xlim(0, T_sweep_h/24)
ax2.grid(alpha=0.3)

# Row 3: histogram of fringe counts + CDF
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(nfr, bins=40, color="#1f4e79", alpha=0.75, edgecolor="white", lw=0.5)
ax3.axvline(stats["nfr_mean"], color="k", lw=2, label=f"mean = {stats['nfr_mean']:.2f}")
ax3.axvline(stats["nfr_median"], color="k", lw=1.2, ls="--",
            label=f"median = {stats['nfr_median']:.2f}")
ax3.axvline(1, color="#c72b2b", ls=":", lw=0.8)
ax3.axvline(3, color="#2e8b57", ls=":", lw=0.8)
ax3.set_xlabel("|fringes|")
ax3.set_ylabel("count (triplets)")
ax3.set_title("Distribution of fringe counts\n(all triplet phases equally likely)", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[2, 1])
srt = np.sort(nfr)
cdf = np.arange(1, len(srt)+1) / len(srt)
ax4.plot(srt, cdf, color="#1f4e79", lw=1.8)
ax4.axvline(1, color="#c72b2b", ls=":", lw=0.8,
            label=f"<1 fr: {stats['frac_below_1fr']*100:.1f}%")
ax4.axvline(3, color="#2e8b57", ls=":", lw=0.8,
            label=f"<3 fr: {stats['frac_below_3fr']*100:.1f}%")
ax4.axvline(5, color="#b279a2", ls=":", lw=0.8,
            label=f"<5 fr: {(1-stats['frac_above_5fr'])*100:.1f}%")
ax4.set_xlabel("|fringes|")
ax4.set_ylabel("CDF")
ax4.set_title("Cumulative distribution of fringe count\n(triplet-phase-averaged)",
              fontsize=10)
ax4.legend(fontsize=8, loc="lower right")
ax4.grid(alpha=0.3)
ax4.set_xlim(0, srt.max())
ax4.set_ylim(0, 1)

# Row 3 col 3: class fraction bar
ax5 = fig.add_subplot(gs[2, 2])
order_cls = ["NULL","WEAK","MARGINAL","GOOD","STRONG","PEAK"]
fracs = [class_fractions.get(c, 0.0)*100 for c in order_cls]
bars = ax5.barh(order_cls, fracs,
                color=[class_colors[c] for c in order_cls],
                edgecolor="k", lw=0.6)
for bar, f in zip(bars, fracs):
    ax5.text(f+0.5, bar.get_y()+bar.get_height()/2, f"{f:.1f}%",
             va="center", fontsize=9)
ax5.set_xlabel("% of sweep")
ax5.set_title("Fraction of triplet phases\nin each fringe class", fontsize=10)
ax5.grid(axis="x", alpha=0.3)
ax5.set_xlim(0, max(fracs)*1.25)
ax5.invert_yaxis()

# Row 4: operational summary text
ax6 = fig.add_subplot(gs[3, :])
ax6.axis("off")
summary = (
    f"OPERATIONAL SUMMARY — rigid triplet sweep over {T_sweep_h/24:.1f}-day lunar month\n"
    f"{'─'*108}\n"
    f"  Of all possible NISAR triplet phases at this site (Thwaites-like tides, A_M2=0.80 m, A_K1=0.50 m):\n\n"
    f"    {stats['frac_null_zone']*100:5.1f}%  NULL zone  (<0.5 fringes)   — no recoverable GL signal\n"
    f"    {stats['frac_below_1fr']*100:5.1f}%  below 1 fringe              — below detectability floor\n"
    f"    {stats['frac_below_3fr']*100:5.1f}%  below 3 fringes             — marginal for GL mapping\n"
    f"    {stats['frac_above_3fr']*100:5.1f}%  ≥ 3 fringes                 — usable for GL mapping\n"
    f"    {stats['frac_above_5fr']*100:5.1f}%  ≥ 5 fringes                 — robust, high-confidence mapping\n\n"
    f"  Fringe count: mean={stats['nfr_mean']:.2f}, median={stats['nfr_median']:.2f}, "
    f"5–95% range = [{stats['nfr_p05']:.2f}, {stats['nfr_p95']:.2f}], max={stats['nfr_max']:.2f}\n"
    f"  h_DD range: [{stats['hDD_min']:+.2f}, {stats['hDD_max']:+.2f}] m over the lunar month"
)
ax6.text(0.01, 0.95, summary, transform=ax6.transAxes,
         family="monospace", fontsize=10.5, va="top",
         bbox=dict(boxstyle="round,pad=0.8", fc="#f5f5f5", ec="#999"))

fig.suptitle("Rigid-triplet phase sweep  —  L-band NISAR, 12-day repeat, 1-hour step over a lunar month",
             fontsize=13, y=0.995)

out_svg = "/home/claude/ddinsar_sweep.svg"
out_png = "/home/claude/ddinsar_sweep.png"
plt.savefig(out_svg, format="svg", bbox_inches="tight")
plt.savefig(out_png, format="png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_svg}")
print(f"Wrote {out_png}")
