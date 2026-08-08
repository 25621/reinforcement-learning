"""A small kitchen the robot can actually fail in, plus the skills it owns.

The point of this file is that "execute the plan" must be able to go WRONG.
A symbolic kitchen where every action always succeeds turns the project into a
language exercise: the language model proposes, the world agrees, and you learn
nothing about robots.  So the kitchen here has geometry.  Objects sit at real
(x, y) positions, the robot has a base position and a finite reach, and a skill
whose target is far away, or crowded by other objects, fails a measurable
fraction of the time.

That gives us the two halves of SayCan honestly:

* the **"say"** half -- which skill is a sensible next step for this sentence --
  which needs world knowledge and comes from a language model;
* the **"can"** half -- whether this robot, standing here, can actually do that
  skill right now -- which is not in any text corpus and has to be measured on
  the robot.

Both halves are scores between 0 and 1, and SayCan multiplies them.
"""

import numpy as np

# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
# Locations are (x, y) in metres on a counter top.  The robot's base sits at
# BASE and its arm can comfortably reach REACH_OK metres; past that the chance
# of a clean grasp falls away, and past REACH_MAX it cannot touch the thing at
# all.
BASE = np.array([0.00, 0.00])
REACH_OK = 0.55
REACH_MAX = 0.85

# name -> (position, is_graspable)
OBJECTS = {
    "mug":         (np.array([0.30, 0.25]), True),
    "kettle":      (np.array([0.52, -0.10]), True),
    "coffee pod":  (np.array([0.18, 0.40]), True),
    "sponge":      (np.array([0.70, 0.35]), True),
    "spoon":       (np.array([0.40, 0.45]), True),
    "counter":     (np.array([0.25, 0.05]), False),
    "sink":        (np.array([0.62, 0.42]), False),
    "trash can":   (np.array([-0.35, 0.55]), False),
}
MOVABLE = [k for k, v in OBJECTS.items() if v[1]]
SURFACES = ["counter", "sink", "trash can"]

# ---------------------------------------------------------------------------
# the skill library
# ---------------------------------------------------------------------------
# Every skill is one English sentence and one Python effect.  This pairing is
# the whole interface between language and control: the language model only
# ever emits one of these sentences, so it can never ask for something the
# robot has no code for.  That is why the skill list is written out by hand
# instead of letting the model free-form -- section 2 of the README measures
# what happens when you let it free-form instead.
PREP = {"counter": "on", "sink": "in", "trash can": "in"}
PLACE = {}                       # skill sentence -> (object, destination)
SKILLS = []
for _o in MOVABLE:
    SKILLS.append(f"pick up the {_o}")
for _o in MOVABLE:
    for _s in SURFACES:
        _sk = f"put the {_o} {PREP[_s]} the {_s}"
        PLACE[_sk] = (_o, _s)
        SKILLS.append(_sk)
SKILLS += [
    "turn on the kettle",
    "pour the kettle into the mug",
    "put the coffee pod in the mug",
    "stir the mug with the spoon",
    "wipe the counter with the sponge",
    "done",
]
SKILLS = list(dict.fromkeys(SKILLS))          # de-duplicate, keep order
N_SKILLS = len(SKILLS)


class Kitchen:
    """Robot state + object state, and one method that runs a skill."""

    def __init__(self, rng, base=None):
        self.rng = rng
        self.base = np.array(BASE if base is None else base, float)
        self.pos = {k: v[0].copy() for k, v in OBJECTS.items()}
        self.held = None            # object in the gripper, or None
        self.kettle_on = False
        self.pod_in_mug = False
        self.mug_filled = False
        self.mug_stirred = False
        self.counter_dirty = True
        self.finished = False
        self.log = []

    # -- geometry -----------------------------------------------------------
    def dist(self, name):
        return float(np.linalg.norm(self.pos[name] - self.base))

    def clutter(self, name):
        """How many other objects sit within 12 cm of this one."""
        p = self.pos[name]
        return sum(1 for k in MOVABLE
                   if k != name and np.linalg.norm(self.pos[k] - p) < 0.12)

    def true_success_prob(self, skill):
        """The kitchen's own answer to 'would this work?'.

        This is the ground truth the affordance model has to LEARN.  It is
        never shown to the planner -- the planner only gets samples of it, the
        same way a real robot only gets trials.
        """
        pre = self.preconditions_met(skill)
        if not pre:
            return 0.0
        tgt = self.skill_target(skill)
        if tgt is None:                       # "done" needs no motion
            return 1.0
        d, c = self.dist(tgt), self.clutter(tgt)
        # Reach: full marks inside REACH_OK, fading to zero at REACH_MAX.
        reach = float(np.clip((REACH_MAX - d) / (REACH_MAX - REACH_OK), 0.0, 1.0))
        p = 0.97 * (0.25 + 0.75 * reach) * (0.88 ** c)
        return float(np.clip(p, 0.0, 1.0))

    # Skills whose reachable target is not the last noun in the sentence: the
    # robot is already holding the tool, so what has to be within reach is the
    # thing being worked on.
    TARGET_OVERRIDE = {
        "stir the mug with the spoon": "mug",
        "wipe the counter with the sponge": "counter",
        "pour the kettle into the mug": "mug",
        "put the coffee pod in the mug": "mug",
    }

    def skill_target(self, skill):
        """The object whose position decides whether the skill is reachable."""
        if skill in self.TARGET_OVERRIDE:
            return self.TARGET_OVERRIDE[skill]
        for name in sorted(OBJECTS, key=len, reverse=True):
            if skill.endswith(name) or f" the {name} " in skill:
                return name
        return None

    # -- symbolic preconditions ---------------------------------------------
    def preconditions_met(self, skill):
        h = self.held
        if skill == "done":
            return True
        if skill.startswith("pick up the "):
            return h is None and skill[len("pick up the "):] in MOVABLE
        if skill in PLACE:
            return h == PLACE[skill][0]
        if skill == "turn on the kettle":
            return not self.kettle_on
        if skill == "pour the kettle into the mug":
            return h == "kettle" and self.kettle_on and not self.mug_filled
        if skill == "put the coffee pod in the mug":
            return h == "coffee pod" and not self.pod_in_mug
        if skill == "stir the mug with the spoon":
            return h == "spoon" and self.mug_filled
        if skill == "wipe the counter with the sponge":
            return h == "sponge" and self.counter_dirty
        return False

    # -- execution ----------------------------------------------------------
    def execute(self, skill):
        """Run one skill.  Returns True if the world actually changed."""
        p = self.true_success_prob(skill)
        ok = self.rng.random() < p
        self.log.append((skill, ok))
        if not ok:
            return False
        if skill == "done":
            self.finished = True
        elif skill == "turn on the kettle":
            self.kettle_on = True
        elif skill == "pour the kettle into the mug":
            self.mug_filled = True
        elif skill == "put the coffee pod in the mug":
            self.pod_in_mug = True
            self.held = None
        elif skill == "stir the mug with the spoon":
            self.mug_stirred = True
        elif skill == "wipe the counter with the sponge":
            self.counter_dirty = False
        elif skill.startswith("pick up the "):
            self.held = skill[len("pick up the "):]
        elif skill in PLACE:
            obj, dest = PLACE[skill]
            self.pos[obj] = self.pos[dest] + self.rng.normal(0, 0.02, 2)
            self.held = None
        return True

    # -- what the planner is allowed to look at -----------------------------
    def describe(self):
        held = self.held if self.held else "nothing"
        bits = [f"The robot is holding {held}."]
        if self.kettle_on:
            bits.append("The kettle is on.")
        if self.pod_in_mug:
            bits.append("There is a coffee pod in the mug.")
        if self.mug_filled:
            bits.append("The mug has hot water in it.")
        if self.mug_stirred:
            bits.append("The mug has been stirred.")
        if not self.counter_dirty:
            bits.append("The counter is clean.")
        return " ".join(bits)

    def features(self, skill):
        """The numbers the LEARNED affordance model sees.

        Deliberately small and deliberately *physical*: distance, clutter, and
        whether the gripper is free.  It does not get told the answer, and it
        does not get told the task -- an affordance is a property of the robot
        and the scene, not of what you asked for.
        """
        tgt = self.skill_target(skill)
        d = self.dist(tgt) if tgt else 0.0
        c = self.clutter(tgt) if tgt else 0.0
        return np.array([d, d * d, c, 1.0 if self.held else 0.0,
                         1.0 if self.preconditions_met(skill) else 0.0, 1.0])


# ---------------------------------------------------------------------------
# the tasks
# ---------------------------------------------------------------------------
# Each task is an English sentence plus a checker.  The checker is the ONLY
# definition of success; nothing scores the plan by how it reads.
TASKS = [
    ("make me a cup of coffee",
     lambda k: k.pod_in_mug and k.mug_filled),
    ("put a coffee pod in the mug",
     lambda k: k.pod_in_mug),
    ("boil some water",
     lambda k: k.kettle_on),
    ("wipe the counter clean",
     lambda k: not k.counter_dirty),
    ("put the sponge in the sink",
     lambda k: np.linalg.norm(k.pos["sponge"] - k.pos["sink"]) < 0.08),
    ("throw the coffee pod in the trash",
     lambda k: np.linalg.norm(k.pos["coffee pod"] - k.pos["trash can"]) < 0.08),
    ("bring the spoon to the counter",
     lambda k: np.linalg.norm(k.pos["spoon"] - k.pos["counter"]) < 0.08),
    ("fill the mug with hot water",
     lambda k: k.mug_filled),
    ("stir the coffee",
     lambda k: k.mug_stirred),
    ("clear the mug off the counter into the sink",
     lambda k: np.linalg.norm(k.pos["mug"] - k.pos["sink"]) < 0.08),
    ("make coffee and stir it",
     lambda k: k.pod_in_mug and k.mug_filled and k.mug_stirred),
    ("tidy up: sponge in the sink and counter wiped",
     lambda k: (not k.counter_dirty)
     and np.linalg.norm(k.pos["sponge"] - k.pos["sink"]) < 0.08),
]


def make_prompt(task, kitchen):
    """The text the language model is asked to continue.

    Two design choices matter and are measured in the README.  First, the
    world state is written into the prompt in plain English -- a language model
    has no sensors, so if you do not tell it the gripper is full it will
    cheerfully propose picking something else up.  Second, the prompt ends
    mid-sentence at "1." so that every candidate skill is scored as the same
    kind of continuation.
    """
    objs = ", ".join(MOVABLE)
    return (
        "A robot is working in a kitchen.\n"
        f"Objects it can move: {objs}.\n"
        f"Places it can use: {', '.join(SURFACES)}.\n"
        f"State: {kitchen.describe()}\n"
        f"Human: {task}\n"
        "Robot, list the next single step:\n"
        "1."
    )
