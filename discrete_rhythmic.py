import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import imageio_ffmpeg

plt.rcParams.update({"font.family": "Arial", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

fs = 500

C_D, C_R = "#f39c12",  "#107181"

# ---------- Discrete: minimum-jerk movements between postures, with dwells ----------
MT, DWELL = 1.5, 1.0                  # movement time, hold time (s)
targets = [0, 1, 0, 1, 0]
targets = [0, 1, 0]
T_total = (MT + DWELL) * (len(targets) - 1) + DWELL
t = np.arange(0, T_total, 1 / fs)

xd = np.zeros_like(t)
dwell_windows = []
t0 = DWELL / 2
dwell_windows.append((0, t0))
for k in range(len(targets) - 1):
    a, b = targets[k], targets[k + 1]
    idx = (t >= t0) & (t < t0 + MT)
    tau = (t[idx] - t0) / MT
    xd[idx] = a + (b - a) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    hold = (t >= t0 + MT) & (t < t0 + MT + DWELL)
    xd[hold] = b
    dwell_windows.append((t0 + MT, t0 + MT + DWELL))
    t0 += MT + DWELL
xd[t >= t0] = targets[-1]

# ---------- Rhythmic: sinusoid, same range and same mean period ----------
f = 1 / (2 * (MT + DWELL))            # one cycle = two discrete movements
xr = 0.5 - 0.5 * np.cos(2 * np.pi * f * t)

def deriv(x):
    v = np.gradient(x, 1 / fs)
    return v, np.gradient(v, 1 / fs)

vd, ad = deriv(xd)
w = 2 * np.pi * f
vr = 0.5 * w * np.sin(w * t)           # analytic derivatives (no edge artifacts)
ar = 0.5 * w**2 * np.cos(w * t)
rev_t = np.arange(0, T_total - 1e-9, 1 / (2 * f))  # reversals (v = 0)

# ---------- Static figure ----------
def static_fig():
    fig, ax = plt.subplots(3, 2, figsize=(12, 6.5), sharex=True)
    for col, (x, v, a, c, title) in enumerate(
            [(xd, vd, ad, C_D, "Discrete"), (xr, vr, ar, C_R, "Rhythmic")]):
        for row, (y, lab) in enumerate([(x, "Position"), (v, "Velocity"), (a, "Acceleration")]):
            axx = ax[row, col]
            axx.set_xlim(0, T_total)
            axx.plot(t, y, color=c, lw=2.5)
            axx.axhline(0, color="0.7", lw=0.8)
            axx.set_yticks([])
            if col == 0:
                axx.set_ylabel(lab)
            if col == 0:
                for w in dwell_windows:
                    axx.axvspan(w[0], min(w[1], T_total), color=C_D, alpha=0.12, lw=0)
            else:
                for tr in rev_t:
                    axx.axvline(tr, color=C_R, ls=":", lw=1)
                if row > 0:
                    i = np.clip((rev_t * fs).astype(int), 0, len(t) - 1)
                    axx.plot(t[i], y[i], "o", color=C_R, ms=7, mfc="white", mew=2)
        ax[0, col].set_title(title, fontweight="bold", color=c)
        ax[2, col].set_xlabel("Time (s)"); ax[2, col].set_xticks([0, 1, 2, 3, 4])
    ax[2, 0].text(0.01, -0.45, "Shaded: postures held, v = 0 and a = 0",
                  transform=ax[2, 0].transAxes, color=C_D, fontsize=11)
    ax[2, 1].text(0.01, -0.45, "Dotted: reversals, v = 0 while |a| is maximal",
                  transform=ax[2, 1].transAxes, color=C_R, fontsize=11)
    fig.tight_layout()
    fig.savefig("discrete_vs_rhythmic.png", dpi=200, transparent=True)
    plt.close(fig)

# ---------- Animation: moving hand + traces building up ----------
def animation():
    for (x, v, a, c, title) in ([(xd, vd, ad, C_D, "Discrete"), (xr, vr, ar, C_R, "Rhythmic")]):
        col = 0
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.6, 1, 1, 1], hspace=0.35)
        step = 10                             # 50 fps
        frames = range(0, len(t), step)
        artists = []    
        top = fig.add_subplot(gs[0, col])
        top.set_xlim(-0.15, 1.15); top.set_ylim(-1, 1); top.axis("off")
        top.plot([0, 1], [0, 0], color="0.75", lw=3)
        top.set_title(title, fontweight="bold", color=c)
        dot, = top.plot([], [], "o", color=c, ms=22)
        lines, heads = [], []
        for row, (y, lab) in enumerate([(x, "Position"), (v, "Velocity"), (a, "Acceleration")]):
            axx = fig.add_subplot(gs[row + 1, col])             # determine the panel
            axx.plot(t, y, color=c, alpha=0.12, lw=2.5)         # ghost of full trace
            axx.axhline(0, color="0.7", lw=0.8)
            axx.set_xlim(0, T_total); axx.set_yticks([])
            pad = 0.1 * np.ptp(y)
            axx.set_ylim(y.min() - pad, y.max() + pad)
            if col == 0: axx.set_ylabel(lab)
            if row < 2: axx.set_xticklabels([])
            else: axx.set_xlabel("Time (s)"); axx.set_xticks(np.arange(0, T_total, 1))
            ln, = axx.plot([], [], color=c, lw=2.5)
            hd, = axx.plot([], [], "o", color=c, ms=6)
            lines.append((ln, y)); heads.append((hd, y))
        artists.append((dot, x, lines, heads))

        def update(i):
            out = []
            for dot, x, lines, heads in artists:
                dot.set_data([x[i]], [0]); out.append(dot)
                for (ln, y), (hd, _) in zip(lines, heads):
                    ln.set_data(t[:i + 1], y[:i + 1]); hd.set_data([t[i]], [y[i]])
                    out += [ln, hd]
            return out

        anim = FuncAnimation(fig, update, frames=frames, blit=True)
        tit = "%s" % title.lower()
        anim.save(tit +".mp4", writer=FFMpegWriter(fps=50, bitrate=3000), dpi=120)
        anim.save(tit +".gif", writer=PillowWriter(fps=25), dpi=80)
        plt.close(fig)

if __name__ == "__main__":
    static_fig()
    animation()
