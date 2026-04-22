"""
3-date DDInSAR simulation at a glacier grounding line.

Geometry:  IFG_A = phi(t2) - phi(t1)      t1=0,  t2=12 days
           IFG_B = phi(t3) - phi(t2)      t2=12, t3=24 days
           DD    = IFG_A - IFG_B
                 = phi(t1) - 2 phi(t2) + phi(t3)

The DD signal therefore depends on tidal loading at ALL THREE epochs:
           h_DD  =  h(t1) - 2 h(t2) + h(t3)

Tide model: realistic mixed semidiurnal + diurnal, constructed from M2 and K1
constituents (periods 12.42 h and 23.93 h respectively). Amplitudes chosen
for a typical Antarctic grounding zone (e.g., Thwaites/Pope/Smith-Kohler:
M2 ~0.8 m, K1 ~0.5 m, background mean 0).

SAR: L-band (lambda = 0.2360 m, NISAR/ALOS-2-like),
     one fringe = lambda/2 * cos(theta) projected to LOS for vertical disp.
     For theta=39 deg: one fringe = 0.118 m / 0.777  ~ 9.2 cm vertical,
     but along-LOS one fringe = lambda/2 = 12 cm EXACTLY (user spec).
     We express fringes in LOS.

Physics: elastic plate flexure, clamped at GL.
  w(s,t) = h(t) * [1 - exp(-beta*s)*(cos(beta*s) + sin(beta*s))], s >= 0
  w = 0 on grounded ice (s<0)
  beta = (rho_w g / 4 D)^(1/4),  D = E H^3 / [12(1-nu^2)]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ===================== Physical parameters =====================
E       = 0.88e9           # Young's modulus of ice (Pa)
nu      = 0.30             # Poisson ratio
H       = 400.0            # ice thickness (m)
rho_w   = 1028.0
g       = 9.81
D       = E * H**3 / (12.0 * (1.0 - nu**2))
beta    = (rho_w * g / (4.0 * D))**0.25
L_flex  = np.pi / beta     # limit of flexure (m)

lam_L   = 0.2360           # L-band wavelength (m)  -> half = 11.8 cm
half_lam = lam_L / 2.0     # one LOS fringe (m)
theta_i = np.deg2rad(39.0)
cos_i   = np.cos(theta_i)

print(f"beta = {beta:.3e} 1/m   L_flex = {L_flex:.0f} m")
print(f"L-band lambda = {lam_L*100:.2f} cm   ->   1 fringe (LOS) = {half_lam*100:.2f} cm")

# ===================== Tide model =====================
# Mixed semidiurnal + diurnal Antarctic-like tide
def tide(t_hours, A_M2=0.80, A_K1=0.50, phi_M2=0.0, phi_K1=1.1):
    T_M2 = 12.42
    T_K1 = 23.93
    return (A_M2 * np.cos(2*np.pi*t_hours/T_M2 + phi_M2)
          + A_K1 * np.cos(2*np.pi*t_hours/T_K1 + phi_K1))

# ===================== Scene geometry =====================
nx, ny = 700, 500
dx     = 15.0  # m / pixel
x = (np.arange(nx) - nx/2) * dx
y = (np.arange(ny) - ny/2) * dx
X, Y = np.meshgrid(x, y)

# Slightly sinuous grounding line
GL_y = 250.0*np.sin(2*np.pi*X/7000.0) + 0.03*X
s    = Y - GL_y  # signed cross-GL distance (>0 = floating)

def flexure(s_grid, h_tide):
    s_pos = np.maximum(s_grid, 0.0)
    w = h_tide * (1.0 - np.exp(-beta*s_pos) * (np.cos(beta*s_pos) + np.sin(beta*s_pos)))
    w[s_grid < 0] = 0.0
    return w

# ===================== Core DDInSAR computation =====================
def simulate_DD(h1, h2, h3, noise=True, seed=1):
    """Return wrapped DD phase, unwrapped DD LOS disp (m), and residual tide."""
    h_DD = h1 - 2*h2 + h3
    w1 = flexure(s, h1); w2 = flexure(s, h2); w3 = flexure(s, h3)
    w_DD = w1 - 2*w2 + w3                        # vertical DD disp (m)
    los_DD = w_DD * cos_i                         # LOS projection
    phase = -4.0*np.pi/lam_L * los_DD             # unwrapped phase (rad)
    wrapped = np.angle(np.exp(1j*phase))
    if noise:
        # realistic coherence: high grounded, lower shelf, low in a shear margin
        coh = 0.88 - 0.22*(s > 0) - 0.35*np.exp(-((X+1500)/400.0)**2 - ((Y-1200)/900.0)**2)
        coh = np.clip(coh, 0.18, 0.95)
        rng = np.random.default_rng(seed)
        sigma = np.sqrt((1 - coh**2) / (2 * coh**2 * 8))  # multilook 8
        wrapped = np.angle(np.exp(1j*(wrapped + rng.normal(0, sigma))))
    return wrapped, los_DD, h_DD

# ===================== Generate multiple triplets =====================
# Sample 6 different triplet start times within a tidal month.
# For each: acquisition times t_start, t_start+12d, t_start+24d.
scenarios = []
# choose start hours that give a variety of h_DD values
start_hours_grid = np.arange(0, 240, 2.0)   # every 2 h over 10 days
hDDs = []
for sh in start_hours_grid:
    h1 = tide(sh)
    h2 = tide(sh + 12*24)
    h3 = tide(sh + 24*24)
    hDDs.append(h1 - 2*h2 + h3)
hDDs = np.array(hDDs)

# pick 6 starts that span the range of |h_DD| from near-zero to large
order = np.argsort(np.abs(hDDs))
picks = [order[2], order[len(order)//6], order[len(order)//3],
         order[len(order)//2], order[3*len(order)//4], order[-1]]
picks_hours = [start_hours_grid[p] for p in picks]

print("\nSelected acquisition start times (h from epoch) and tidal loads:")
for sh in picks_hours:
    h1 = tide(sh); h2 = tide(sh+12*24); h3 = tide(sh+24*24)
    print(f"  t0={sh:5.1f} h  ->  h1={h1:+.3f}  h2={h2:+.3f}  h3={h3:+.3f}   "
          f"h_DD={h1-2*h2+h3:+.3f} m   |fringes| ~ "
          f"{abs((h1-2*h2+h3)*cos_i)/half_lam:.1f}")

# ===================== Plot: grid of 6 DD interferograms =====================
fig = plt.figure(figsize=(16, 11), facecolor="white")
gs = fig.add_gridspec(3, 3, hspace=0.70, wspace=0.25,
                     height_ratios=[1, 1, 0.95])

cmap = plt.get_cmap("hsv")

# --- Top row: tide time series with triplet markers ---
ax_tide = fig.add_subplot(gs[0, :])
t_plot = np.linspace(0, 24*24 + 48, 4000)
ax_tide.plot(t_plot/24, tide(t_plot), color="#1f4e79", lw=1.3, label="Ocean tide (M2+K1)")
ax_tide.axhline(0, color="k", lw=0.4)
ax_tide.set_xlabel("Time (days)")
ax_tide.set_ylabel("Tide height (m)")
ax_tide.set_title("Realistic tide signal (mixed semidiurnal + diurnal, A$_{M2}$=0.80 m, A$_{K1}$=0.50 m) "
                  "and six chosen 3-date triplets (12-day repeat)",
                  fontsize=12)
colors = plt.get_cmap("tab10").colors
for i, sh in enumerate(picks_hours):
    for k, dt in enumerate([0, 12*24, 24*24]):
        t_acq = sh + dt
        h_acq = tide(t_acq)
        ax_tide.plot(t_acq/24, h_acq, "o", color=colors[i], ms=8,
                     mec="k", mew=0.8, zorder=5)
        ax_tide.annotate(f"S{i+1}.{k+1}", (t_acq/24, h_acq),
                         xytext=(3, 6), textcoords="offset points",
                         fontsize=8, color=colors[i], fontweight="bold")
ax_tide.grid(alpha=0.3)
ax_tide.set_xlim(-0.5, t_plot.max()/24 + 1)

# --- Middle + bottom rows: 6 DD interferograms ---
axes_ddi = [fig.add_subplot(gs[1 + i//3, i%3]) for i in range(6)]
for i, (sh, ax) in enumerate(zip(picks_hours, axes_ddi)):
    h1 = tide(sh); h2 = tide(sh + 12*24); h3 = tide(sh + 24*24)
    wrapped, los_DD, h_DD = simulate_DD(h1, h2, h3, seed=100+i)

    im = ax.imshow(wrapped,
                   extent=[x.min()/1000, x.max()/1000, y.min()/1000, y.max()/1000],
                   origin="lower", cmap=cmap, vmin=-np.pi, vmax=np.pi,
                   interpolation="bilinear")

    # GL + flexure limit
    x_line = np.linspace(x.min(), x.max(), 300)
    gl_line = 250.0*np.sin(2*np.pi*x_line/7000.0) + 0.03*x_line
    ax.plot(x_line/1000, gl_line/1000, color="white", lw=1.5)
    ax.plot(x_line/1000, gl_line/1000, color="k", lw=0.7, linestyle="--")
    ax.plot(x_line/1000, (gl_line+L_flex)/1000, color="white", lw=0.8,
            linestyle=(0,(3,3)))

    # total fringe count estimate from h_DD
    n_fringes = abs(h_DD * cos_i) / half_lam
    title = (f"S{i+1}   t$_0$ = {sh:.0f} h   (tide triplet)\n"
             f"h$_1$={h1:+.2f} m,  h$_2$={h2:+.2f} m,  h$_3$={h3:+.2f} m\n"
             f"h$_{{DD}}$ = {h_DD:+.2f} m   →   ≈ {n_fringes:.1f} fringes")
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_xlabel("Range (km)", fontsize=9)
    ax.set_ylabel("Azimuth (km)", fontsize=9)
    ax.tick_params(labelsize=8)

    # coloured border to link to tide plot
    for spine in ax.spines.values():
        spine.set_edgecolor(colors[i]); spine.set_linewidth(2.5)

# shared colorbar on the right
cbar_ax = fig.add_axes([0.93, 0.08, 0.014, 0.50])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(f"DD wrapped phase (rad)  —  1 fringe = λ/2 = {half_lam*100:.1f} cm LOS",
               fontsize=10)
cbar.set_ticks([-np.pi, 0, np.pi]); cbar.set_ticklabels([r"$-\pi$","0",r"$\pi$"])

fig.suptitle("3-date Double-Difference InSAR at a Grounding Line — L-band (λ=23.6 cm), 12-day repeat\n"
             r"DD$=\phi_{t_1}-2\phi_{t_2}+\phi_{t_3}$   |   h$_{DD}=h_1-2h_2+h_3$   |   "
             f"H={int(H)} m, E={E/1e9:.2f} GPa, L$_{{flex}}$≈{L_flex/1000:.1f} km",
             fontsize=13, y=0.995)

out_svg = "/home/claude/ddinsar_3date_Lband.svg"
out_png = "/home/claude/ddinsar_3date_Lband.png"
plt.savefig(out_svg, format="svg", bbox_inches="tight")
plt.savefig(out_png, format="png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nWritten: {out_svg}")
print(f"Written: {out_png}")

# ===================== CSV summary of scenarios =====================
import csv
csv_path = "/home/claude/ddinsar_scenarios.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario","t0_hours","h1_m","h2_m","h3_m","h_DD_m",
                "LOS_DD_peak_m","fringes_peak","L_flex_m","notes"])
    for i, sh in enumerate(picks_hours):
        h1 = tide(sh); h2 = tide(sh+12*24); h3 = tide(sh+24*24)
        hDD = h1 - 2*h2 + h3
        los_peak = abs(hDD) * cos_i
        nfr = los_peak / half_lam
        note = ("near-null: triplet insensitive to tide" if nfr < 0.5
                else "weak fringe pattern" if nfr < 2
                else "good for GL mapping")
        w.writerow([f"S{i+1}", f"{sh:.1f}", f"{h1:+.3f}", f"{h2:+.3f}",
                    f"{h3:+.3f}", f"{hDD:+.3f}", f"{los_peak:.4f}",
                    f"{nfr:.2f}", f"{L_flex:.0f}", note])
print(f"Written: {csv_path}")

# ===================== Transect comparison figure =====================
fig2, (axT, axP) = plt.subplots(2, 1, figsize=(11, 7),
                                 gridspec_kw={"height_ratios":[1,1], "hspace":0.35},
                                 facecolor="white")

# perpendicular-ish transect at range = 0
col = nx // 2
s_trans = s[:, col]
order = np.argsort(s_trans)

# Top: flexure profiles (ground truth)
for i, sh in enumerate(picks_hours):
    h1 = tide(sh); h2 = tide(sh+12*24); h3 = tide(sh+24*24)
    hDD = h1 - 2*h2 + h3
    w1 = flexure(s_trans, h1); w2 = flexure(s_trans, h2); w3 = flexure(s_trans, h3)
    w_DD = w1 - 2*w2 + w3
    axT.plot(s_trans[order]/1000, w_DD[order]*100,
             color=colors[i], lw=2,
             label=f"S{i+1}: h$_{{DD}}$={hDD:+.2f} m")
axT.axvline(0, color="k", ls="--", lw=1)
axT.axvline(L_flex/1000, color="grey", ls=(0,(3,3)), lw=1)
axT.text(0.05, axT.get_ylim()[1]*0.88 if axT.get_ylim()[1]>0 else -2,
         "F (GL)", fontsize=9, fontweight="bold")
axT.text(L_flex/1000+0.05, axT.get_ylim()[1]*0.88 if axT.get_ylim()[1]>0 else -2,
         "H (limit of flexure)", fontsize=9, fontweight="bold")
axT.set_xlim(-3, 5.5)
axT.axhline(0, color="k", lw=0.4)
axT.set_xlabel("Distance from GL (km)")
axT.set_ylabel("DD vertical displacement (cm)")
axT.set_title("DD flexure profiles across the grounding zone — 6 triplet scenarios")
axT.legend(loc="center right", fontsize=8, ncol=1)
axT.grid(alpha=0.3)

# Bottom: peak LOS disp vs h_DD (linearity check)
hDD_scan = np.array([tide(sh) - 2*tide(sh+12*24) + tide(sh+24*24)
                     for sh in start_hours_grid])
peak_los = np.abs(hDD_scan) * cos_i
axP.plot(hDD_scan*100, peak_los*100, '.', color="#1f4e79", alpha=0.4, ms=4,
         label="all possible triplet start times (2 h grid)")
# overlay the 6 picks
for i, sh in enumerate(picks_hours):
    h1 = tide(sh); h2 = tide(sh+12*24); h3 = tide(sh+24*24)
    hDD = h1 - 2*h2 + h3
    axP.plot(hDD*100, abs(hDD)*cos_i*100, 'o', color=colors[i], ms=10,
             mec="k", mew=0.8, label=f"S{i+1}")
# fringe count gridlines
for nfr in [1,2,3,5,7]:
    hcrit = nfr * half_lam / cos_i
    axP.axhline(nfr*half_lam*100, color="grey", lw=0.4, ls=":")
    axP.text(axP.get_xlim()[1]*0.95, nfr*half_lam*100+0.6,
             f"{nfr} fringes", fontsize=8, ha="right", color="grey")
axP.set_xlabel("h$_{DD}$ = h$_1$ - 2h$_2$ + h$_3$  (cm)")
axP.set_ylabel("Peak LOS displacement (cm)")
axP.set_title("DD sensitivity: observable vs tidal triplet residual  —  L-band, 1 fringe = 11.8 cm LOS")
axP.legend(loc="upper center", fontsize=8, ncol=4)
axP.grid(alpha=0.3)

out_svg2 = "/home/claude/ddinsar_3date_transects.svg"
out_png2 = "/home/claude/ddinsar_3date_transects.png"
plt.savefig(out_svg2, format="svg", bbox_inches="tight")
plt.savefig(out_png2, format="png", dpi=140, bbox_inches="tight")
plt.close(fig2)
print(f"Written: {out_svg2}")
print(f"Written: {out_png2}")
