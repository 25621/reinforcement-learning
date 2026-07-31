"""A minimal URDF parser -- about 120 lines of real work.

URDF ("Unified Robot Description Format") is just XML.  A robot is a list of
LINKS (rigid bodies) and a list of JOINTS, where each joint names one parent
link, one child link, a fixed offset between them, and an axis to move about.
That is the whole format for our purposes; the rest of a production URDF is
meshes, collision volumes, materials and Gazebo plug-ins.

What this file deliberately does NOT do is compute forward kinematics.  Parsing
("what is this robot made of?") and kinematics ("where are its parts right
now?") are different jobs, and keeping them apart is what lets project 03
rebuild the kinematics from scratch and check it against an outside reference
while still reading the same robot description.
"""

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import numpy as np


# ---------------------------------------------------------------------------
# small XML helpers
# ---------------------------------------------------------------------------
def _vec(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def _rpy_to_R(rpy):
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix: Rz(yaw) Ry(pitch) Rx(roll).

    Duplicated from project 01's ``transforms.py`` on purpose, so that this
    parser has no dependency outside the standard library plus NumPy.
    """
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(elem):
    """Read an ``<origin xyz=... rpy=.../>`` child into a 4x4 transform."""
    T = np.eye(4)
    o = None if elem is None else elem.find("origin")
    if o is not None:
        T[:3, :3] = _rpy_to_R(_vec(o.get("rpy"), [0, 0, 0]))
        T[:3, 3] = _vec(o.get("xyz"), [0, 0, 0])
    return T


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class Visual:
    """One drawable shape, positioned by ``T`` inside its own link's frame."""

    kind: str  # 'cylinder' | 'box' | 'sphere'
    params: dict
    T: np.ndarray


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)
    mass: float = 0.0


@dataclass
class Joint:
    name: str
    jtype: str  # 'revolute' | 'continuous' | 'prismatic' | 'fixed'
    parent: str
    child: str
    T_origin: np.ndarray  # fixed pose of the child frame when the joint is at 0
    axis: np.ndarray  # unit axis, expressed in the CHILD frame
    lower: float
    upper: float

    @property
    def movable(self):
        return self.jtype != "fixed"


class Robot:
    """A parsed URDF: links, joints, and the tree that connects them."""

    def __init__(self, name, links, joints):
        self.name = name
        self.links = {l.name: l for l in links}
        self.joints = joints
        self.child_to_joint = {j.child: j for j in joints}
        self.children = {l.name: [] for l in links}
        for j in joints:
            self.children[j.parent].append(j.child)

        # The root is the one link that is nobody's child.
        roots = [n for n in self.links if n not in self.child_to_joint]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, found {roots}")
        self.root = roots[0]

        # Order joints so that a parent always comes before its children.  This
        # is what lets forward kinematics be a single left-to-right sweep with
        # no recursion and no re-visiting.  Children are visited in file order,
        # so the joint ordering matches what other URDF loaders produce.
        self.ordered = []

        def _walk(link):
            for c in self.children[link]:
                self.ordered.append(self.child_to_joint[c])
                _walk(c)

        _walk(self.root)

        self.movable = [j for j in self.ordered if j.movable]
        self.joint_names = [j.name for j in self.movable]
        self.n = len(self.movable)
        self.lower = np.array([j.lower for j in self.movable])
        self.upper = np.array([j.upper for j in self.movable])

    # -- convenience ------------------------------------------------------
    def random_q(self, rng, n=None):
        """Uniform samples INSIDE the joint limits."""
        size = (self.n,) if n is None else (n, self.n)
        return rng.uniform(self.lower, self.upper, size=size)

    def clamp(self, q):
        return np.clip(q, self.lower, self.upper)

    def limit_margin(self, q):
        """Distance to the nearest limit, per joint (negative = violated)."""
        return np.minimum(q - self.lower, self.upper - q)

    def tree_str(self):
        """An ASCII drawing of the kinematic tree, the way ``tree`` prints files."""
        lines = []

        def rec(link, indent):
            lines.append(f"{indent}{link}")
            for c in self.children[link]:
                j = self.child_to_joint[c]
                xyz = " ".join(f"{v:g}" for v in j.T_origin[:3, 3])
                if j.movable:
                    ax = " ".join(f"{v:g}" for v in j.axis)
                    tail = f"axis=({ax})  limits=[{j.lower:+.2f}, {j.upper:+.2f}]"
                else:
                    tail = "no degree of freedom"
                lines.append(f"{indent}  |")
                lines.append(f"{indent}  +-- <{j.name}> {j.jtype}: xyz=({xyz})  {tail}")
            for c in self.children[link]:
                rec(c, indent + "      ")

        rec(self.root, "")
        return "\n".join(lines)

    def summary(self):
        n_fixed = sum(1 for j in self.joints if not j.movable)
        return (
            f"robot '{self.name}': {len(self.links)} links, {len(self.joints)} joints "
            f"({self.n} movable + {n_fixed} fixed), root '{self.root}'"
        )


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------
def load_urdf(path):
    root = ET.parse(path).getroot()

    links = []
    for le in root.findall("link"):
        visuals = []
        for ve in le.findall("visual"):
            g = ve.find("geometry")
            T = _origin(ve)
            if g.find("cylinder") is not None:
                c = g.find("cylinder")
                visuals.append(
                    Visual("cylinder", {"radius": float(c.get("radius")), "length": float(c.get("length"))}, T)
                )
            elif g.find("box") is not None:
                visuals.append(Visual("box", {"size": _vec(g.find("box").get("size"), [0.1, 0.1, 0.1])}, T))
            elif g.find("sphere") is not None:
                visuals.append(Visual("sphere", {"radius": float(g.find("sphere").get("radius"))}, T))
        ie = le.find("inertial")
        mass = 0.0 if ie is None or ie.find("mass") is None else float(ie.find("mass").get("value"))
        links.append(Link(le.get("name"), visuals, mass))

    joints = []
    for je in root.findall("joint"):
        jtype = je.get("type")
        axis = _vec(je.find("axis").get("xyz") if je.find("axis") is not None else None, [0, 0, 1])
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        lim = je.find("limit")
        if jtype == "continuous" or lim is None:
            # A 'continuous' joint spins forever, so URDF gives it no limits.
            lower, upper = -np.pi, np.pi
        else:
            lower, upper = float(lim.get("lower")), float(lim.get("upper"))
        joints.append(
            Joint(
                name=je.get("name"),
                jtype=jtype,
                parent=je.find("parent").get("link"),
                child=je.find("child").get("link"),
                T_origin=_origin(je),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )

    return Robot(root.get("name"), links, joints)
