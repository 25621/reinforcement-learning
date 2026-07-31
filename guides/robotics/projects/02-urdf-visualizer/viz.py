"""3-D drawing helpers for a parsed URDF.

The renderer takes POSES as an argument -- it never computes kinematics itself.
That is what makes it reusable: project 02 hands it poses from a 12-line sweep,
project 03 hands it poses from a verified module, and project 03's bug study
hands it poses from a deliberately broken one, all through the same call.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FRAME_COLORS = ("#D55E00", "#009E73", "#0072B2")  # x = red, y = green, z = blue


def _apply(T, P):
    """Transform an (..., 3) array of points by a 4x4 transform."""
    return P @ T[:3, :3].T + T[:3, 3]


def set_axes_equal(ax, radius=None, center=None):
    """Force equal scaling on all three axes.

    Matplotlib's 3-D axes stretch each axis independently by default, which
    turns a round arm into an ellipse and makes a right angle look wrong.  A
    robot picture with unequal axes is actively misleading, so every figure
    here calls this.
    """
    if center is None or radius is None:
        lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        center = lims.mean(axis=1)
        radius = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def draw_frame(ax, T, scale=0.09, lw=1.6, alpha=1.0):
    """Draw a coordinate triad: red = x, green = y, blue = z."""
    o = T[:3, 3]
    for i, c in enumerate(FRAME_COLORS):
        d = T[:3, i] * scale
        ax.plot(*zip(o, o + d), color=c, lw=lw, alpha=alpha)


def _cylinder(ax, T, radius, length, color, alpha, n=18):
    th = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([radius * np.cos(th), radius * np.sin(th), np.zeros(n)], axis=1)
    bot = _apply(T, circle + np.array([0, 0, -length / 2]))
    top = _apply(T, circle + np.array([0, 0, length / 2]))
    X = np.stack([bot[:, 0], top[:, 0]])
    Y = np.stack([bot[:, 1], top[:, 1]])
    Z = np.stack([bot[:, 2], top[:, 2]])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)
    for ring in (bot, top):
        ax.add_collection3d(Poly3DCollection([ring], color=color, alpha=alpha, linewidths=0))


def _sphere(ax, T, radius, color, alpha, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2 + 1)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    P = _apply(T, np.stack([x, y, z], axis=-1).reshape(-1, 3)).reshape(x.shape + (3,))
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=color, alpha=alpha, linewidth=0, shade=True)


def _box(ax, T, size, color, alpha):
    sx, sy, sz = np.asarray(size) / 2.0
    c = np.array(
        [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ]
    )
    c = _apply(T, c)
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([c[f] for f in faces], color=color, alpha=alpha, linewidths=0))


def draw_robot(
    ax,
    robot,
    poses,
    color="#7FA8C9",
    alpha=0.95,
    bone_color="#2B3A46",
    show_frames=(),
    frame_scale=0.09,
    show_bones=True,
):
    """Draw a robot given ``poses``: a dict mapping link name -> 4x4 world pose.

    ``show_frames`` is a list of link names whose coordinate triads to draw.
    """
    if show_bones:
        for j in robot.joints:
            a, b = poses[j.parent][:3, 3], poses[j.child][:3, 3]
            if np.linalg.norm(a - b) > 1e-9:
                ax.plot(*zip(a, b), color=bone_color, lw=2.2, alpha=min(1.0, alpha + 0.05), zorder=1)

    for name, link in robot.links.items():
        for v in link.visuals:
            T = poses[name] @ v.T
            if v.kind == "cylinder":
                _cylinder(ax, T, v.params["radius"], v.params["length"], color, alpha)
            elif v.kind == "sphere":
                _sphere(ax, T, v.params["radius"], color, alpha)
            elif v.kind == "box":
                _box(ax, T, v.params["size"], color, alpha)

    for name in show_frames:
        draw_frame(ax, poses[name], scale=frame_scale)


def draw_ground(ax, radius=0.9, color="#DDDDDD", z=0.0):
    th = np.linspace(0, 2 * np.pi, 40)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th), np.full(40, z)], axis=1)
    ax.add_collection3d(Poly3DCollection([ring], color=color, alpha=0.35, linewidths=0))


def style_3d(ax, title=None, radius=0.75, center=(0, 0, 0.45), elev=20, azim=-60, ticks=True):
    ax.view_init(elev=elev, azim=azim)
    set_axes_equal(ax, radius=radius, center=np.asarray(center, dtype=float))
    if ticks:
        ax.set_xlabel("x (m)", labelpad=-8)
        ax.set_ylabel("y (m)", labelpad=-8)
        ax.set_zlabel("z (m)", labelpad=-8)
        ax.tick_params(labelsize=6, pad=-3)
    else:
        ax.set_axis_off()
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
    if title:
        ax.set_title(title, fontsize=9)
