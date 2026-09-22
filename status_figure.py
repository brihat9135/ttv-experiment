"""
Project-status figure (the "where we stand" graphic, distinct from the science-result
blob figure). Renders a roadmap arc with a YOU-ARE-HERE marker plus the headline finding
and its 16%->83% near-resonance mass-tightening inset. Writes status_figure.png.

Honest status: done milestones are filled/green, open ones are hollow/grey.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

DONE = "#2e7d32"      # green  = completed
PEND = "#9aa0a6"      # grey   = not yet done
HERE = "#1f6fb2"      # blue   = you-are-here / result accent

# (label, sublabel, done?)
MILESTONES = [
    ("PoC",            "2-param MDN,\ncalibrated, fast",        True),
    ("Full 6-param",   "theta=(m1,m2,\nh1,k1,h2,k2)",           True),
    ("Resonance\ndiagnosis", "collapse is\nphysical, not a\nmodel defect", True),
    ("Observables",    "durations pin h,\nRV pins k + mass",    True),
    ("Flow head",      "MDN -> norm. flow\n(fix training\ninstability)", False),
    ("jaxttv\nbaseline", "parity vs the\nincumbent HMC",        False),
    ("Real-data\nvalidation", "Kepler-9,\nTRAPPIST-1,\nSBC at scale", False),
    ("Survey scale",   "TESS / PLATO\ncatalogs; tool\n+ paper", False),
]
HERE_IDX = 3          # last completed milestone

fig = plt.figure(figsize=(13.5, 7.0))
ax = fig.add_axes([0.03, 0.30, 0.94, 0.60])
ax.set_xlim(-0.6, len(MILESTONES) - 0.4)
ax.set_ylim(-1.7, 1.9)
ax.axis("off")

# connecting spine
ax.plot([0, len(MILESTONES) - 1], [0, 0], color="#cfcfcf", lw=3, zorder=1)

for i, (label, sub, done) in enumerate(MILESTONES):
    col = DONE if done else PEND
    face = col if done else "white"
    ax.scatter([i], [0], s=560, facecolor=face, edgecolor=col, linewidth=2.6, zorder=3)
    ax.text(i, 0, "✓" if done else "…", ha="center", va="center",
            color="white" if done else col, fontsize=15, fontweight="bold", zorder=4)
    # title above, detail below, alternating to avoid crowding
    ax.text(i, 0.42, label, ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#222" if done else "#555")
    ax.text(i, -0.42, sub, ha="center", va="top", fontsize=8.2, color="#555")
    if not done and (i == 0 or MILESTONES[i - 1][2]):
        ax.text(i, 1.02, "next", ha="center", va="bottom", fontsize=8.5,
                style="italic", color=HERE)

# YOU ARE HERE marker
ax.annotate("YOU ARE HERE", xy=(HERE_IDX, 0.24), xytext=(HERE_IDX, 1.45),
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=HERE,
            arrowprops=dict(arrowstyle="-|>", color=HERE, lw=2.4))

# divider between done and pending
xd = HERE_IDX + 0.5
ax.axvline(xd, ymin=0.12, ymax=0.88, color="#dddddd", ls=(0, (4, 4)), lw=1.2)
ax.text(HERE_IDX / 2.0, -1.5, "BUILT + VALIDATED (simulation)", ha="center",
        fontsize=9.5, color=DONE, fontweight="bold")
ax.text((HERE_IDX + len(MILESTONES)) / 2.0, -1.5, "REMAINING", ha="center",
        fontsize=9.5, color=PEND, fontweight="bold")

# ---- headline finding (bottom-left text panel) ----
axf = fig.add_axes([0.03, 0.02, 0.62, 0.24]); axf.axis("off")
axf.text(0, 1.0,
         "Finding", fontsize=12.5, fontweight="bold", color="#222", va="top")
axf.text(0, 0.72,
         "Near resonance the mass-eccentricity posterior is a physically degenerate\n"
         "ridge (data-limited, not a model defect). The right observables break it:\n"
         "durations pin h, radial velocity pins k and mass directly. The bottleneck\n"
         "is observational, not algorithmic.",
         fontsize=10, color="#333", va="top", linespacing=1.5)

# ---- inset: 16% -> 83% near-resonance mass tightening ----
axb = fig.add_axes([0.72, 0.045, 0.24, 0.205])
bars = axb.bar([0, 1], [16, 83], color=[PEND, HERE], width=0.62)
axb.set_ylim(0, 100)
axb.set_xticks([0, 1])
axb.set_xticklabels(["timing\n+durations", "+RV"], fontsize=8.5)
axb.set_ylabel("m2 tightening (%)", fontsize=8.5)
axb.set_title("Near-resonance mass\ndegeneracy broken", fontsize=9, fontweight="bold")
for b, v in zip(bars, [16, 83]):
    axb.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
             fontsize=9, fontweight="bold")
axb.spines[["top", "right"]].set_visible(False)
axb.tick_params(labelsize=8)

fig.suptitle("TTV mass inference  —  project status",
             fontsize=15.5, fontweight="bold", x=0.5, y=0.975)

fig.savefig("status_figure.png", dpi=145)
print("wrote status_figure.png")
