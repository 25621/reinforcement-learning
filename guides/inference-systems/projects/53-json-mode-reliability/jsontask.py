"""The extraction task both project 53 and project 56 run on.

Kept in its own module so the two projects share one schema, one regular
expression and one set of test cases. If 56 re-declared them, a later tweak
to the schema would silently make the two projects' numbers incomparable.
"""

from __future__ import annotations

import random

# The schema, twice: once as a JSON Schema (what a user writes) and once as
# the regular expression it compiles to (what the decoder can actually use).
# Turning the first into the second is the entire job of a library like
# Outlines; here it is written out by hand so both halves are visible.
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "skills"],
}

PATTERN = (r'\{"name": "[A-Za-z ]+", "age": (0|[1-9][0-9]*), '
           r'"skills": \["[a-z ]+"(, "[a-z ]+")*\]\}')

INSTRUCTION = ('Return a JSON object with keys "name" (string), "age" '
               '(integer) and "skills" (array of strings). Output only JSON.')

SYSTEM = "You output JSON and nothing else."

FIRST = ["Ada", "Grace", "Alan", "Katherine", "Linus", "Barbara", "Dennis",
         "Radia", "Tim", "Margaret", "Ken", "Frances", "Donald", "Shafi"]
LAST = ["Lovelace", "Hopper", "Turing", "Johnson", "Torvalds", "Liskov",
        "Ritchie", "Perlman", "Berners", "Hamilton", "Thompson", "Allen"]
SKILLS = ["mathematics", "compilers", "networking", "cryptography",
          "databases", "graphics", "robotics", "linguistics", "optics",
          "statistics", "avionics", "typography"]


def make_cases(n: int, seed: int = 11):
    """Synthetic bios whose ground truth we know, so accuracy is checkable.

    Real captions would make the accuracy column a judgement call; here the
    right answer is whatever the generator put in, so "name correct" is a
    string comparison and not an opinion.
    """
    rnd = random.Random(seed)
    cases = []
    for _ in range(n):
        name = f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"
        age = rnd.randrange(21, 89)
        sk = rnd.sample(SKILLS, rnd.randrange(1, 4))
        bio = (f"{name} is {age} years old. "
               f"{name.split()[0]} works on {', '.join(sk)}.")
        cases.append({"name": name, "age": age, "skills": sk, "bio": bio})
    return cases
