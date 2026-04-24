# Physics

`tidal-insar-sim` assembles four well-established ingredients:

1. Harmonic tide synthesis from CATS2008 constituents (via `pyTMD`).
2. Elastic-plate flexure of an ice shelf clamped at the grounding line.
3. The double-difference (DD) geometry between three SAR acquisitions.
4. Translation of the DD displacement into DDInSAR fringes along the
   line-of-sight.

Every formula below is referenced to peer-reviewed literature.

## 1. Tide forcing

At a site $(\\lambda, \\phi)$ we synthesise the tide from the local
harmonic constants extracted from CATS2008 (Padman et al., 2002; 2018):

$$
h(t) = \\sum_{i} A_i \\cos\\!\\big(\\omega_i t - \\varphi_i\\big),
$$

where $A_i$ are amplitudes and $\\varphi_i$ are Greenwich phases of the
$i$-th tidal constituent at the point. Default constituents are the four
semidiurnal (M2, S2, N2, K2) and four diurnal (K1, O1, P1, Q1) harmonics.
`pyTMD.compute.tide_elevations` applies the 18.6-year nodal correction
and the astronomical equilibrium formula.

## 2. Elastic-plate flexure

The floating ice shelf is modelled as a thin elastic plate clamped at the
grounding line, following Rignot (2011), Fricker and Padman (2006), and
Walker et al. (2013):

$$
D = \\frac{E H^3}{12(1-\\nu^2)}, \\qquad
\\beta = \\left(\\frac{\\rho_w g}{4 D}\\right)^{1/4}.
$$

The vertical deflection at cross-GL distance $s$ (with $s \\ge 0$ floating
and $s < 0$ grounded) under a uniform water load $h(t)$ is

$$
w(s, t) = \\begin{cases}
h(t)\\left[1 - e^{-\\beta s}(\\cos \\beta s + \\sin \\beta s)\\right] & s \\ge 0, \\\\
0 & s < 0.
\\end{cases}
$$

The "limit of flexure" $L_\\text{flex} = \\pi / \\beta$ marks the peak of
the forebulge, where $w(L_\\text{flex}) / h = 1 + e^{-\\pi} \\approx 1.043$.

**Validated numerics** (from `reference/prototype_v1_three_date.py`; $H
= 400$ m, $E = 0.88$ GPa, $\\nu = 0.30$): $\\beta = 8.362 \\times 10^{-4}$
m$^{-1}$ and $L_\\text{flex} = 3757$ m, matched by our implementation to
better than 0.1%.

## 3. Double-difference geometry

Given three SAR acquisitions at times $t_1, t_2, t_3$ with temporal
baseline $B = t_2 - t_1 = t_3 - t_2$ (e.g. 12 days for NISAR), the two
successive interferograms $\\phi_A = \\phi(t_2) - \\phi(t_1)$ and
$\\phi_B = \\phi(t_3) - \\phi(t_2)$ are differenced to cancel the
steady-state flow component:

$$
\\phi_{DD}(x) = \\phi_A - \\phi_B = \\phi(t_1) - 2\\phi(t_2) + \\phi(t_3).
$$

The same structure holds for the vertical displacement:

$$
h_{DD}(x) = w(x, t_1) - 2 w(x, t_2) + w(x, t_3).
$$

Near the grounding line, most of the secular signal cancels; what remains
is the tidally driven flexure residual.

## 4. Fringe count

Projecting the vertical DD displacement onto the radar line-of-sight and
converting to fringes:

$$
\\phi_{DD}(x) = -\\frac{4\\pi}{\\lambda}\\cos\\theta\\; h_{DD}(x),
\\qquad
n_\\text{fringes,peak} = \\frac{\\lvert h_{DD}^\\text{max}\\rvert \\cos\\theta}{\\lambda / 2}.
$$

One LOS fringe corresponds to $\\lambda/2$ of displacement. For NISAR L-band
($\\lambda = 23.6$ cm, $\\theta = 39^\\circ$), one fringe = 11.8 cm along
LOS and $\\approx 15.2$ cm vertical.

## 5. Rigid-triplet phase sweep

For operational planning we sweep the *entire* triplet through an arbitrary
offset $\\Delta t$ (keeping $B$ fixed) over one synodic month (29.53 d) at
1-h resolution. At each $\\Delta t$ we compute $h_{DD}$ and $n_\\text{fringes,peak}$,
then reduce the distribution to:

- $P(\\ge 3 \\text{fr})$ — fraction "usable" for GL mapping.
- $P(\\ge 5 \\text{fr})$ — fraction "robust".
- $P(< 0.5 \\text{fr})$ — fraction in the "null zone".

These probabilities drive the verdict tier (excellent / good / marginal / poor).

## 6. Confidence under acquisition jitter

The per-triplet `confidence` column in the planner output is

$$
\\text{confidence} = 1 - P\\!\\big(n_\\text{fringes} < 1 \\ \\big|\\ \\delta t_i \\sim U[-1, +1]\\ \\text{h}\\big),
$$

where $\\delta t_i$ are independent uniform jitters at each of the three
acquisitions, estimated by a 4 000-realisation Monte Carlo.

A "rock-solid" triplet has confidence = 1 (every jittered realisation still
clears the detectability floor); a "brittle" triplet near a null has low
confidence even though its nominal fringe count is high.

## References

- Fricker, H. A., and Padman, L. (2006). *GRL* 33, L15502.
- Padman, L., Fricker, H. A., Coleman, R., Howard, S., Erofeeva, L. (2002).
  *Annals of Glaciology* 34, 247–254.
- Padman, L., Siegfried, M. R., Fricker, H. A. (2018). *Reviews of
  Geophysics* 56, doi:10.1002/2016RG000546.
- Rignot, E., Mouginot, J., Scheuchl, B. (2011). *GRL* 38, L10504.
- Walker, R. T., Christianson, K., Parizek, B. R., Anandakrishnan, S.,
  Alley, R. B. (2013). *EPSL* 395, 184–193.
- Milillo, P., et al. (2019). *Science Advances* 5, eaau3433.
