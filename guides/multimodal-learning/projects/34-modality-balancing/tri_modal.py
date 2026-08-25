"""The tri-modal corpus: faces, captions, and spoken digits in one alphabet.

Two things live here, and project 35 imports both.

Part 1 -- the third modality: spoken digits turned into whole numbers by EnCodec.

Project 32 built an image tokenizer from scratch. For audio we do NOT rebuild
one -- a neural audio codec is exactly the same idea (encoder, codebook,
decoder) applied to a waveform, and Phase 6 project 28 already took one apart.
Here we simply *use* Meta's pretrained `facebook/encodec_24khz` as a frozen
tokenizer, which is what a real unified model does too: image and audio
tokenizers are trained separately and then frozen, and only the transformer on
top is trained jointly.

Output: 64 whole numbers per clip, drawn from a 1024-entry codebook -- the same
count as project 32's 64 image tokens, on purpose, so that any imbalance we
measure later comes from the *data mixture* and not from one modality happening
to be more verbose than another.

Why only EnCodec's first codebook. EnCodec is *residual*: codebook 1 encodes the
waveform, codebook 2 encodes what codebook 1 got wrong, and so on. Taking the
first alone is the coarsest, most speech-like layer, and it keeps the sequence
short. Reconstruction from it alone is rough, but this project never decodes
audio -- it only measures how a joint model divides its attention.

Part 2 -- `build()` and the three row builders, which put faces, spoken digits
and their captions into ONE shared alphabet and ONE sequence format.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-mel-spectrogram-pipeline"))
sys.path.insert(0, str(HERE.parent / "32-discrete-image-tokens"))
sys.path.insert(0, str(HERE.parent / "33-tiny-chameleon"))
CTX = 88                       # 1 + 20 caption words + 1 + 64 codes + 2 markers

ENCODEC = "facebook/encodec_24khz"
SR = 24000
FRAMES = 64                    # 64 / 75 fps = 0.853 s of audio
SAMPLES = FRAMES * 320         # EnCodec's stride is 320 samples per frame
AUDIO_CODES = 1024             # entries in EnCodec's first codebook
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five",
               "six", "seven", "eight", "nine"]


def data_dir():
    return HERE / "data"


def _load_fsdd():
    """Spoken digits at 24 kHz, from Phase 2's shared loader."""
    import audio_lib as A
    xs, digits, speakers = A.load_fsdd(HERE.parent / "06-mel-spectrogram-pipeline" / "data",
                                       sr_out=SR)
    # A.load_fsdd pads every clip to 1.024 s; we keep the first 0.853 s
    xs = np.stack([np.pad(x, (0, max(0, SAMPLES - len(x))))[:SAMPLES] for x in xs])
    return xs.astype(np.float32), digits, speakers


def build_audio_tokens(verbose=True):
    """(N, 64) int16 EnCodec codes, digit labels, speaker ids. Cached."""
    data_dir().mkdir(parents=True, exist_ok=True)
    cache = data_dir() / "audio_tokens.npz"
    if cache.exists():
        z = np.load(cache)
        return z["codes"], z["digits"], z["speakers"]

    from transformers import EncodecModel
    xs, digits, speakers = _load_fsdd()
    model = EncodecModel.from_pretrained(ENCODEC).eval()
    codes = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(xs), 32):
            wav = torch.from_numpy(xs[i:i + 32])[:, None]        # (B, 1, T)
            enc = model.encode(wav, bandwidth=1.5)
            # enc.audio_codes: (chunks, B, n_codebooks, frames) -- keep codebook 0
            c = enc.audio_codes[0][:, 0, :FRAMES]
            codes.append(c.to(torch.int16).numpy())
            if verbose and i % 512 == 0:
                print(f"  encoded {i}/{len(xs)}", flush=True)
    codes = np.concatenate(codes)
    if verbose:
        print(f"  {len(codes)} clips -> {codes.shape[1]} tokens each "
              f"in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, codes=codes, digits=digits, speakers=speakers)
    return codes, digits, speakers


def audio_caption(digit, speaker_id):
    """The text that goes with a clip: 'the spoken digit five'."""
    return f"the spoken digit {DIGIT_WORDS[int(digit)]}"


# ---------------------------------------------------------------------------
# the tri-modal corpus
# ---------------------------------------------------------------------------
HELD_OUT_SPEAKER = 5            # 'yweweler', the same voice Phase 2 held out


def build():
    """(bundle of arrays, shared Vocab). Cached under 34/data/corpus.npz."""
    import unified as U
    import vqvae as VQ
    data_dir().mkdir(parents=True, exist_ok=True)
    cache = data_dir() / "corpus.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    else:
        p = U.load_pairs()
        acodes, digits, speakers = build_audio_tokens()
        acaps = np.array([audio_caption(dg, sp) for dg, sp in zip(digits, speakers)])
        # hold out one speaker, exactly as Phase 2 did -- a random split would
        # put the same voice saying the same digit on both sides
        held = speakers == HELD_OUT_SPEAKER
        d = dict(img_tr=p["tr_codes"], cap_tr=p["tr_caps"],
                 img_va=p["va_codes"], cap_va=p["va_caps"],
                 aud_tr=acodes[~held], acap_tr=acaps[~held],
                 aud_va=acodes[held], acap_va=acaps[held])
        np.savez_compressed(cache, **d)
    vocab = U.build_vocab(
        list(d["cap_tr"]) + list(d["cap_va"]) + list(d["acap_tr"]) + list(d["acap_va"]),
        VQ.CODEBOOK, AUDIO_CODES)
    return d, vocab


def image_rows(vocab, caps, codes):
    """<bos> caption <boi> 64 image codes <eoi> <eos>"""
    import unified as U
    return [[U.BOS] + vocab.text_ids(c)[:U.TEXT_CTX] + [U.BOI]
            + vocab.image_ids(ic) + [U.EOI, U.EOS] for c, ic in zip(caps, codes)]


def audio_rows(vocab, caps, codes):
    """<bos> caption <boa> 64 audio codes <eoa> <eos>"""
    import unified as U
    return [[U.BOS] + vocab.text_ids(c)[:U.TEXT_CTX] + [U.BOA]
            + vocab.audio_ids(ac) + [U.EOA, U.EOS] for c, ac in zip(caps, codes)]


def text_rows(vocab, caps):
    """<bos> caption <eos>"""
    import unified as U
    return [[U.BOS] + vocab.text_ids(c)[:U.TEXT_CTX] + [U.EOS] for c in caps]


def mixed_val(vocab, d):
    """One validation set containing both kinds of row, so every arm is scored
    on exactly the same tokens."""
    import unified as U
    rows = (image_rows(vocab, d["cap_va"], d["img_va"])
            + audio_rows(vocab, d["acap_va"], d["aud_va"]))
    return U.pad_batch(rows, CTX)
