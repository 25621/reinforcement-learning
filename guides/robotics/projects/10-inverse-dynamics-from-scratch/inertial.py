"""Read the mass properties out of a URDF.

Project 02's parser stops at "what shapes should I draw, and how are the links
connected?", because that is all kinematics ever needs -- where a link *is*
does not depend on how heavy it is.  Dynamics is the first place where mass
matters, so this file adds the missing half of the description: for every link,
its mass, the position of its centre of mass, and its 3x3 inertia tensor; and
for every joint, its torque limit.

Keeping this separate from ``urdf.py`` rather than editing it is deliberate.
Project 02's parser is already used by four projects, and a parser that only
reads what it needs is easier to trust.  This module takes a ``Robot`` that has
already been parsed and hangs the extra numbers off it, so nothing downstream
of project 02 changes.

URDF's ``<inertia>`` gives six numbers because an inertia tensor is symmetric:
ixx, iyy, izz on the diagonal and ixy, ixz, iyz off it.  The tensor is
expressed in a frame that sits AT the centre of mass, oriented by the
``<origin rpy>`` of the ``<inertial>`` block.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Inertial:
    """Mass properties of one link, in that link's own frame."""

    mass: float
    com: np.ndarray  # 3-vector: centre of mass in the link frame
    I: np.ndarray  # 3x3 inertia tensor ABOUT THE CENTRE OF MASS, link-frame axes

    @property
    def is_massless(self):
        return self.mass <= 0.0


def _vec(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def _rpy_to_R(rpy):
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


def load_inertials(path):
    """link name -> :class:`Inertial`, read straight from the URDF XML."""
    root = ET.parse(path).getroot()
    out = {}
    for le in root.findall("link"):
        name = le.get("name")
        ie = le.find("inertial")
        if ie is None:
            out[name] = Inertial(0.0, np.zeros(3), np.zeros((3, 3)))
            continue

        mass = float(ie.find("mass").get("value"))

        o = ie.find("origin")
        com = _vec(None if o is None else o.get("xyz"), [0, 0, 0])
        R_com = _rpy_to_R(_vec(None if o is None else o.get("rpy"), [0, 0, 0]))

        e = ie.find("inertia")
        ixx, iyy, izz = float(e.get("ixx")), float(e.get("iyy")), float(e.get("izz"))
        ixy, ixz, iyz = float(e.get("ixy")), float(e.get("ixz")), float(e.get("iyz"))
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        # If the <inertial> origin carries a rotation, the tensor is written in
        # that rotated frame.  Rotate it into the link frame so every tensor in
        # the model speaks the same axes:  I_link = R I_com R^T.
        if not np.allclose(R_com, np.eye(3)):
            I = R_com @ I @ R_com.T

        out[name] = Inertial(mass, com, I)
    return out


def load_effort_limits(path):
    """movable joint name -> torque (or force) limit in N*m (or N).

    URDF calls this ``effort`` because the same field means a torque on a
    revolute joint and a force on a prismatic one -- "effort" is the word that
    covers both.
    """
    root = ET.parse(path).getroot()
    out = {}
    for je in root.findall("joint"):
        if je.get("type") == "fixed":
            continue
        lim = je.find("limit")
        out[je.get("name")] = float(lim.get("effort")) if lim is not None else np.inf
    return out


def load_velocity_limits(path):
    """movable joint name -> speed limit in rad/s (or m/s)."""
    root = ET.parse(path).getroot()
    out = {}
    for je in root.findall("joint"):
        if je.get("type") == "fixed":
            continue
        lim = je.find("limit")
        out[je.get("name")] = float(lim.get("velocity")) if lim is not None else np.inf
    return out
