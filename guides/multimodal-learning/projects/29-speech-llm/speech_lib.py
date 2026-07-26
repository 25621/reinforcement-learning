"""Data for a speech LLM: two-digit utterances, encoded once by frozen Whisper.

The model itself is project 20's `vlm_lib.py` with one substitution -- frozen
CLIP becomes frozen Whisper -- which is the whole point of this project, and the
reason there is no model code here.

What this file provides:

  1. `build_utterances()` glues two FSDD digit recordings into one 2-second clip,
     so the answer is a *sequence* ("four seven") and not a 10-way choice. A
     model that has learned only the answer format cannot fake a sequence: pure
     guessing scores 1 in 100.
  2. `build_cache()` runs the frozen Whisper encoder over every utterance once
     and stores two taps -- the last layer (what speech LLMs actually use) and
     an early layer -- so "which layer should the projector read?" is a
     measurement rather than an assumption.
  3. `AudioLLMData` serves pooled audio tokens plus question/answer pairs for
     two tasks: *what was said* and *who said it*.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "06-mel-spectrogram-pipeline"))
import audio_lib as A  # noqa: E402

WHISPER = "openai/whisper-tiny"
SR = 16000
SLOT = 16384                     # 1.024 s per digit, FSDD's own clip length
N_DIGITS = 2                     # digits per utterance
VALID_FRAMES = 104               # 2.048 s x 50 encoder frames/s, rounded up
POOL = 4                         # 104 frames -> 26 audio tokens
AUDIO_DIM = 384                  # whisper-tiny's encoder width
N_UTTER = 2000
WORDS = ["zero", "one", "two", "three", "four", "five",
         "six", "seven", "eight", "nine"]
SPEAKER_NAMES = {"george": "george", "jackson": "jackson", "lucas": "lucas",
                 "nicolas": "nicolas", "theo": "theo", "yweweler": "yweweler"}
HELD_OUT_SPEAKER = "yweweler"


def data_dir():
    return HERE / "data"


# ---------------------------------------------------------------------------
def build_utterances(n=N_UTTER, seed=0):
    """Two random digit recordings from the same speaker, back to back.

    Same speaker, because a voice changing mid-utterance is a cue no real
    recording has, and a model can use it to find the boundary for free.
    """
    cache = data_dir() / "utterances.npz"
    if cache.exists():
        z = np.load(cache)
        return z["wav"], z["digits"], z["spk"]
    xs, digits, spk = A.load_fsdd(PROJECTS / "06-mel-spectrogram-pipeline" / "data",
                                  sr_out=SR)
    rng = np.random.default_rng(seed)
    by_spk = {s: np.where(spk == s)[0] for s in np.unique(spk)}
    wav = np.zeros((n, SLOT * N_DIGITS), dtype=np.float32)
    dig = np.zeros((n, N_DIGITS), dtype=np.int64)
    who = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s = int(rng.integers(0, len(by_spk)))
        picks = rng.choice(by_spk[s], N_DIGITS, replace=False)
        for k, p in enumerate(picks):
            wav[i, k * SLOT:(k + 1) * SLOT] = xs[p]
            dig[i, k] = digits[p]
        who[i] = s
    data_dir().mkdir(parents=True, exist_ok=True)
    np.savez(cache, wav=wav, digits=dig, spk=who)
    return wav, dig, who


def whisper_encoder(name=WHISPER):
    """Whisper's *encoder* only, frozen.

    Why only the encoder: Whisper's decoder is a text model that already knows
    how to write transcripts, and reusing it would answer a different question
    ("is Whisper good at ASR?"). A speech LLM keeps the ears and throws away the
    mouth, then borrows a general-purpose LLM's mouth instead -- one that can
    also answer questions, follow instructions and reason, which Whisper's
    transcription-only decoder cannot.
    """
    from transformers import WhisperModel, WhisperProcessor
    torch.set_num_threads(12)
    pr = WhisperProcessor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name, dtype=torch.float32).encoder.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return pr, enc


@torch.no_grad()
def build_cache(n=N_UTTER, batch=32, verbose=True):
    """Encode every utterance once. Two taps, both truncated to the real audio.

    Truncation matters: Whisper pads every input to 30 seconds, so 1,396 of the
    1,500 encoder frames for our 2-second clip describe silence. Project 07
    measured the cost of mean-pooling all of them -- 17 accuracy points -- and
    here it would also mean 1,500 audio tokens instead of 26.
    """
    last_path = data_dir() / "feat_last.npy"
    if last_path.exists():
        return
    wav, _, _ = build_utterances(n)
    pr, enc = whisper_encoder()
    out = {k: np.zeros((n, VALID_FRAMES, AUDIO_DIM), dtype=np.float16)
           for k in ("last", "early")}
    t0 = time.time()
    for j in range(0, n, batch):
        chunk = wav[j:j + batch]
        f = pr.feature_extractor(list(chunk), sampling_rate=SR,
                                 return_tensors="pt").input_features
        res = enc(f, output_hidden_states=True)
        out["last"][j:j + len(chunk)] = res.last_hidden_state[:, :VALID_FRAMES].numpy()
        out["early"][j:j + len(chunk)] = res.hidden_states[1][:, :VALID_FRAMES].numpy()
        if verbose and j % (batch * 20) == 0:
            print(f"    encoded {j}/{n} ({time.time() - t0:.0f}s)", flush=True)
    np.save(data_dir() / "feat_early.npy", out["early"])
    np.save(last_path, out["last"])           # written last: the "done" marker
    if verbose:
        print(f"    cache built in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
QUESTIONS = {
    "digits": "What digits are spoken?",
    "speaker": "Who is speaking?",
}


class AudioLLMData:
    """Pooled audio tokens + the two question/answer tasks.

    Splits:
      train       five voices, most utterances
      seen_voice  five voices, held-out utterances  (both tasks are answerable)
      new_voice   the sixth voice, never trained on (only the digit task is
                  answerable -- nobody can name a speaker they have never met,
                  so scoring the speaker task there would measure nothing)
    """

    def __init__(self, n=N_UTTER, layer="last", pool=POOL, n_held=300, seed=0):
        build_cache(n)
        self.wav, self.digits, self.spk = build_utterances(n)
        self.feats = np.load(data_dir() / f"feat_{layer}.npy", mmap_mode="r")[:n]
        self.pool = pool
        held = A.SPEAKERS.index(HELD_OUT_SPEAKER)
        rng = np.random.default_rng(seed)
        self.new_voice = np.where(self.spk == held)[0]
        rest = rng.permutation(np.where(self.spk != held)[0])
        self.seen_voice, self.train_ids = rest[:n_held], rest[n_held:]

    def n_tokens(self):
        return VALID_FRAMES // self.pool

    def audio_tokens(self, ids):
        """Average neighbouring encoder frames into one token each.

        26 tokens for 2 seconds of speech is still four times the rate of the
        text it describes. Pooling is the cheapest knob a speech LLM has: the
        LLM's cost grows with the square of the sequence length, and silence and
        steady vowels do not need 50 vectors a second.
        """
        x = torch.from_numpy(np.asarray(self.feats[np.asarray(ids)],
                                        dtype=np.float32))
        b, t, d = x.shape
        return x.reshape(b, t // self.pool, self.pool, d).mean(2)

    def qa(self, i, task):
        if task == "digits":
            return QUESTIONS[task], " ".join(WORDS[d] for d in self.digits[i])
        return QUESTIONS[task], A.SPEAKERS[self.spk[i]]

    def answer_digits(self, text):
        got = [w for w in text.strip().lower().split() if w in WORDS]
        return got[:N_DIGITS]
