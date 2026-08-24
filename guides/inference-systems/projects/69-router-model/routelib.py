"""Shared pieces for project 69 — a tiny model that routes traffic.

Two serving paths, both real and both on this CPU:

  fast path  SmolLM2-135M-Instruct   (cheap, often good enough)
  slow path  Qwen2.5-1.5B-Instruct   (11x the parameters, 11x the bill)

and a set of 60 questions with a checkable answer, deliberately mixed so that
some are trivial and some are not.  The router's job is to send each question to
the cheapest path that will still get it right.

The grader is exact: a numeric answer must match the number, a string answer
must appear in the completion.  No judge model, no partial credit — a router
experiment is only as trustworthy as its labels.
"""

from __future__ import annotations

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FAST_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SLOW_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-0.5B-Instruct"       # the prompted router
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_THREADS = 6

SYSTEM = "Answer with the final answer only. Be brief."

# (category, question, answer, kind)  kind: "num" or "str"
QUERIES = [
    # ---- easy factual (a 135M model can do these) -------------------------
    ("fact", "What is the capital city of France?", "paris", "str"),
    ("fact", "What is the capital city of Japan?", "tokyo", "str"),
    ("fact", "What is the capital city of Italy?", "rome", "str"),
    ("fact", "What is the capital city of Egypt?", "cairo", "str"),
    ("fact", "What colour is the sky on a clear day?", "blue", "str"),
    ("fact", "How many days are in a week?", 7, "num"),
    ("fact", "How many hours are in a day?", 24, "num"),
    ("fact", "How many legs does a spider have?", 8, "num"),
    ("fact", "What is the largest ocean on Earth?", "pacific", "str"),
    ("fact", "Which planet is closest to the Sun?", "mercury", "str"),
    ("fact", "What language is mainly spoken in Brazil?", "portuguese", "str"),
    ("fact", "How many sides does a triangle have?", 3, "num"),

    # ---- easy arithmetic --------------------------------------------------
    ("arith", "What is 12 + 15?", 27, "num"),
    ("arith", "What is 9 * 6?", 54, "num"),
    ("arith", "What is 100 - 37?", 63, "num"),
    ("arith", "What is 81 divided by 9?", 9, "num"),
    ("arith", "What is 7 * 8?", 56, "num"),
    ("arith", "What is 250 + 250?", 500, "num"),
    ("arith", "What is half of 46?", 23, "num"),
    ("arith", "What is 10 percent of 90?", 9, "num"),
    ("arith", "What is 3 to the power of 3?", 27, "num"),
    ("arith", "What is 45 + 55?", 100, "num"),
    ("arith", "What is 15 * 3?", 45, "num"),
    ("arith", "What is 144 divided by 12?", 12, "num"),

    # ---- extraction: the answer is in the prompt --------------------------
    ("extract", "Text: 'Grace Hopper, age 45, works on compilers.' "
                "What is her age? Answer with a number.", 45, "num"),
    ("extract", "Text: 'Order 8842 ships on Tuesday.' "
                "What is the order number?", 8842, "num"),
    ("extract", "Text: 'The meeting moved from room 12 to room 19.' "
                "Which room is the meeting in now?", 19, "num"),
    ("extract", "Text: 'Ada paid 30 euros and Linus paid 12 euros.' "
                "How many euros did Linus pay?", 12, "num"),
    ("extract", "Text: 'Flight AZ204 departs at gate 7.' "
                "What is the gate number?", 7, "num"),
    ("extract", "Text: 'The file weighs 512 kilobytes.' "
                "How many kilobytes?", 512, "num"),
    ("extract", "Text: 'Katherine Johnson worked at NASA.' "
                "Which organisation did she work at?", "nasa", "str"),
    ("extract", "Text: 'The package arrives in Berlin on Friday.' "
                "Which city?", "berlin", "str"),
    ("extract", "Text: 'Barbara Liskov received the Turing Award.' "
                "Which award?", "turing", "str"),
    ("extract", "Text: 'Temperature dropped to -4 degrees overnight.' "
                "What was the temperature?", -4, "num"),
    ("extract", "Text: 'Invoice total is 1290 dollars.' "
                "What is the total in dollars?", 1290, "num"),
    ("extract", "Text: 'The server is in the Dublin region.' "
                "Which region?", "dublin", "str"),

    # ---- multi-step word problems (this is where 135M falls over) ---------
    ("multi", "A shop sells pens for 3 dollars and notebooks for 7 dollars. "
              "Tom buys 4 pens and 3 notebooks. How much does he pay?",
     33, "num"),
    ("multi", "A train travels 60 km in 45 minutes. At the same speed, how "
              "many km does it travel in 2 hours?", 160, "num"),
    ("multi", "Alice is twice as old as Bob. In 5 years the sum of their ages "
              "will be 40. How old is Alice now?", 20, "num"),
    ("multi", "A rectangle is 3 times as long as it is wide and its perimeter "
              "is 64 cm. What is its area in square cm?", 192, "num"),
    ("multi", "A shirt costs 40 dollars, is discounted 25 percent, then the "
              "new price rises 20 percent. What is the final price?", 36,
     "num"),
    ("multi", "A tank holds 480 litres. A pipe adds 8 litres per minute while "
              "a leak removes 3. How many minutes to fill it?", 96, "num"),
    ("multi", "In a class of 30 students, 2/5 play football and half of the "
              "rest play chess. How many play chess?", 9, "num"),
    ("multi", "A car uses 7 litres per 100 km and fuel costs 2 dollars per "
              "litre. What is the fuel cost of a 350 km trip in dollars?",
     49, "num"),
    ("multi", "Machine A makes 12 parts per hour and machine B makes 18. They "
              "work together 3 hours, then A alone for 2 more hours. How many "
              "parts in total?", 114, "num"),
    ("multi", "A book has 240 pages. Sam reads 1/4 on Monday, 1/3 of the rest "
              "on Tuesday, then 40 pages. How many pages are left?", 80,
     "num"),
    ("multi", "A worker earns 18 dollars an hour for 40 hours and 1.5x that "
              "rate beyond 40. She worked 47 hours. What is her pay?", 909,
     "num"),
    ("multi", "Three friends split a 45 dollar bill. Ann pays twice what Ben "
              "pays and Cal pays 5 more than Ben. How much does Ann pay?",
     20, "num"),

    # ---- knowledge that needs more than 135M parameters -------------------
    ("hard_fact", "Who wrote the play 'Romeo and Juliet'?", "shakespeare",
     "str"),
    ("hard_fact", "What gas do plants absorb from the air for photosynthesis?",
     "carbon dioxide", "str"),
    ("hard_fact", "In which country is the city of Marrakesh?", "morocco",
     "str"),
    ("hard_fact", "What is the chemical symbol for gold?", "au", "str"),
    ("hard_fact", "Who developed the theory of general relativity?",
     "einstein", "str"),
    ("hard_fact", "What is the longest river in South America?", "amazon",
     "str"),
    ("hard_fact", "How many minutes are in three and a half hours?", 210,
     "num"),
    ("hard_fact", "What is the boiling point of water in Fahrenheit?", 212,
     "num"),
    ("hard_fact", "In what year did the Berlin Wall fall?", 1989, "num"),
    ("hard_fact", "How many bits are in four bytes?", 32, "num"),
    ("hard_fact", "What is the square root of 169?", 13, "num"),
    ("hard_fact", "How many continents are there?", 7, "num"),
]


def load(model_id: str, threads: int = N_THREADS):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    m.eval()
    return tok, m


def chat_ids(tok, user: str, system: str = SYSTEM) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def generate(model, tok, id_lists, max_new=32, batch=16):
    """Greedy batch generation. Returns (texts, wall_seconds, total_new_tokens)."""
    texts, n_tok, t0 = [], 0, time.time()
    for i in range(0, len(id_lists), batch):
        chunk = id_lists[i:i + batch]
        width = max(len(x) for x in chunk)
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (width - len(x)) + x for x in chunk])
        att = torch.tensor([[0] * (width - len(x)) + [1] * len(x)
                            for x in chunk])
        gen = model.generate(input_ids=inp, attention_mask=att,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad)
        for row in gen[:, width:].tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t == tok.eos_token_id:
                    cut = j
                    break
            n_tok += cut
            texts.append(tok.decode(row[:cut]))
    return texts, time.time() - t0, n_tok


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def graded(text: str, answer, kind: str) -> bool:
    if kind == "num":
        hits = _NUM.findall(text.replace("$", " ").replace(",", ""))
        for h in hits:                     # any number in the reply may be it
            try:
                if abs(float(h) - float(answer)) < 1e-6:
                    return True
            except ValueError:
                pass
        return False
    return str(answer).lower() in text.lower()


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])
