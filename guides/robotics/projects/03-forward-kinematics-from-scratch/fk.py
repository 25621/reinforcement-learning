"""Forward kinematics, written out and verified.

Forward kinematics answers: *given the joint values, where is every part of the
robot?*  It is one sweep down the tree, and each step is the same two-part
move -- go through the joint's FIXED offset, then through its MOVING part:

    T_world_child  =  T_world_parent @ T_origin(joint) @ T_move(joint, q_i)

``T_origin`` comes from the URDF and never changes.  ``T_move`` is a rotation
about the joint axis (revolute) or a slide along it (prismatic), both expressed
in the CHILD frame.  Everything else in this phase -- Jacobians, inverse
kinematics, null-space control, hand-eye calibration -- calls this function.

The module also carries a set of DELIBERATELY BROKEN variants (``BUGS``).  They
exist so project 03 can measure what each classic frame mistake actually costs,
instead of describing it in words.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01-transform-calculator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02-urdf-visualizer"))

import transforms as tf  # noqa: E402


# ---------------------------------------------------------------------------
# the moving part of one joint
# ---------------------------------------------------------------------------
def joint_transform(joint, value):
    """The 4x4 transform contributed by ONE joint at ONE joint value.

    Revolute  -> rotate ``value`` radians about the joint axis.
    Prismatic -> slide ``value`` metres along the joint axis.
    Fixed     -> identity (a fixed joint carries no joint value at all).
    """
    T = np.eye(4)
    if joint.jtype == "prismatic":
        T[:3, 3] = joint.axis * value
    elif joint.jtype in ("revolute", "continuous"):
        T[:3, :3] = tf.axis_angle_to_R(joint.axis * value)
    return T


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def fk_all(robot, q):
    """World pose of EVERY link.  Returns ``{link_name: 4x4}``.

    ``robot.ordered`` guarantees a parent is always processed before its
    children, so one pass is enough -- no recursion, nothing revisited.  The
    cost is one 4x4 matrix product per joint, which is why forward kinematics
    is always cheap and always has an answer (unlike inverse kinematics).
    """
    poses = {robot.root: np.eye(4)}
    qi = 0
    for j in robot.ordered:
        value = 0.0
        if j.movable:
            value = q[qi]
            qi += 1
        poses[j.child] = poses[j.parent] @ j.T_origin @ joint_transform(j, value)
    return poses


def fk(robot, q, link="tool0"):
    """World pose of one link -- the call the rest of the phase actually makes."""
    return fk_all(robot, q)[link]


def fk_position(robot, q, link="tool0"):
    return fk_all(robot, q)[link][:3, 3]


def joint_axes_world(robot, q):
    """For each movable joint: its world-frame origin and its world-frame axis.

    Project 04's analytic Jacobian is built entirely from these two things, so
    it is worth computing them in the same sweep that computes the poses.
    """
    poses = fk_all(robot, q)
    out = []
    for j in robot.movable:
        T = poses[j.child]
        out.append((T[:3, 3].copy(), T[:3, :3] @ j.axis))
    return out, poses


# ---------------------------------------------------------------------------
# Deliberately broken variants, for the bug study
# ---------------------------------------------------------------------------
def fk_all_buggy(robot, q, bug):
    """Same sweep, with exactly one classic mistake switched on."""
    poses = {robot.root: np.eye(4)}
    qi = 0
    for j in robot.ordered:
        value = 0.0
        if j.movable:
            value = q[qi]
            qi += 1

        T_origin = j.T_origin
        T_move = joint_transform(j, value)

        if bug == "transposed_rotation":
            # Rotation matrices are not symmetric: R.T is the rotation BACKWARDS.
            T_move[:3, :3] = T_move[:3, :3].T
        elif bug == "swapped_order":
            # Turning first and then applying the offset is a different robot.
            poses[j.child] = poses[j.parent] @ T_move @ T_origin
            continue
        elif bug == "unnormalised_axis":
            # Use the axis exactly as written in the file.  If the file says
            # "0 2 0", the joint turns twice as far as it should.
            raw = j.axis * getattr(j, "_axis_norm", 1.0)
            T_move = np.eye(4)
            if j.jtype == "prismatic":
                T_move[:3, 3] = raw * value
            elif j.movable:
                T_move[:3, :3] = tf.axis_angle_to_R(raw * value)
        elif bug == "rpy_reversed":
            # Read <origin rpy> as Rx Ry Rz instead of URDF's Rz Ry Rx.
            T_origin = _rpy_reversed_origin(j)
        elif bug == "no_tool_offset":
            # "The tool is at the wrist, near enough."
            if not j.movable:
                poses[j.child] = poses[j.parent]
                continue

        poses[j.child] = poses[j.parent] @ T_origin @ T_move
    return poses


def _rpy_reversed_origin(j):
    """Rebuild a joint origin with the roll-pitch-yaw multiplication reversed."""
    rpy = tf.R_to_rpy(j.T_origin[:3, :3])
    R_wrong = tf.Rx(rpy[0]) @ tf.Ry(rpy[1]) @ tf.Rz(rpy[2])
    return tf.T_from_Rp(R_wrong, j.T_origin[:3, 3])


BUGS = {
    "transposed_rotation": "the joint rotation is transposed (R.T instead of R)",
    "swapped_order": "the joint moves BEFORE its fixed offset instead of after",
    "unnormalised_axis": "the axis from the file is used without normalising it",
    "rpy_reversed": "rpy is read as Rx.Ry.Rz instead of URDF's Rz.Ry.Rx",
    "no_tool_offset": "the fixed tool offset is skipped",
}


def annotate_axis_norms(robot, path):
    """Record each joint's RAW axis length, so the 'forgot to normalise' bug
    can reproduce what a sloppy loader would have done."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    raw = {}
    for je in root.findall("joint"):
        ae = je.find("axis")
        if ae is not None:
            v = np.array([float(x) for x in ae.get("xyz").split()])
            raw[je.get("name")] = float(np.linalg.norm(v))
    for j in robot.joints:
        j._axis_norm = raw.get(j.name, 1.0)
    return robot
