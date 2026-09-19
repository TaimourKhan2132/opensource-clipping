"""
Snap AI-chosen clip boundaries to real speech boundaries.

The AI returns start/end times from reading a timestamped transcript, and it is
often a second or two off — clips then start mid-word or end while the speaker
is still talking. These helpers move each boundary onto the word-level
transcript: clip starts land on a sentence start, clip ends land after the
sentence the speaker is in the middle of, and smart-trim segment edges land on
word edges.
"""

SENTENCE_END = (".", "!", "?", "…", '."', '!"', '?"')
PAUSE_GAP = 0.6          # a silence this long counts as a sentence break
MAX_EXTEND_END = 10.0    # never extend a clip end by more than this
MAX_EXTEND_START = 2.5   # never pull a clip start back by more than this
LEAD_IN = 0.12           # seconds kept before the first word
TAIL = 0.45              # seconds kept after the last word


def _flatten_words(data_segmen):
    words = []
    for seg in data_segmen or []:
        for w in seg.get("words", []):
            try:
                words.append({"word": str(w["word"]), "start": float(w["start"]), "end": float(w["end"])})
            except (KeyError, TypeError, ValueError):
                continue
    words.sort(key=lambda w: w["start"])
    return words


def _is_annotation(word):
    """Caption sound tags such as (Applause), [Music], (Laughter)."""
    return word["word"].lstrip().startswith(("(", "["))


def _is_break_after(words, i):
    """True if a sentence ends after word i (punctuation, sound tag or a long pause)."""
    if words[i]["word"].rstrip().endswith(SENTENCE_END) or _is_annotation(words[i]):
        return True
    if i + 1 < len(words) and _is_annotation(words[i + 1]):
        return True
    return i + 1 < len(words) and words[i + 1]["start"] - words[i]["end"] >= PAUSE_GAP


def _snap_start(words, t, sentence=True):
    idx = next((i for i, w in enumerate(words) if w["end"] > t), None)
    if idx is None:
        return t
    if sentence:
        j = idx
        while j > 0 and not _is_break_after(words, j - 1) and t - words[j - 1]["start"] <= MAX_EXTEND_START:
            j -= 1
        if j == 0 or _is_break_after(words, j - 1):
            idx = j
    prev_end = words[idx - 1]["end"] if idx > 0 else 0.0
    return max(0.0, prev_end + 0.02, words[idx]["start"] - LEAD_IN)


def _snap_end(words, t, sentence=True):
    idx = None
    for i, w in enumerate(words):
        if w["start"] < t:
            idx = i
        else:
            break
    if idx is None:
        return t
    if sentence:
        # If only a sliver (<1s) of a new sentence slipped in, cut back to the break
        b = idx - 1
        while b >= 0 and not _is_break_after(words, b) and words[idx]["end"] - words[b]["start"] < 1.0:
            b -= 1
        if b >= 0 and _is_break_after(words, b) and words[idx]["end"] - words[b + 1]["start"] < 1.0:
            idx = b
        j = idx
        while not _is_break_after(words, j) and j + 1 < len(words) and words[j + 1]["end"] - t <= MAX_EXTEND_END:
            j += 1
        if _is_break_after(words, j):
            idx = j
    next_start = words[idx + 1]["start"] if idx + 1 < len(words) else float("inf")
    word_end = words[idx]["end"]
    if _is_annotation(words[idx]):
        word_end = min(word_end, words[idx]["start"] + 1.0)  # keep ~1s of applause/laughter
    return max(min(word_end + TAIL, next_start - 0.05), min(word_end, next_start - 0.02))


def snap_clip_boundaries(clips, data_segmen):
    """Adjust start/end (and keep_segments) of every clip in place; returns clips."""
    words = _flatten_words(data_segmen)
    if not words:
        return clips

    for clip in clips:
        try:
            old_start, old_end = float(clip["start_time"]), float(clip["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        new_start = _snap_start(words, old_start)
        new_end = _snap_end(words, old_end)
        if new_end <= new_start:
            continue
        clip["start_time"], clip["end_time"] = round(new_start, 3), round(new_end, 3)

        segs = clip.get("keep_segments")
        if isinstance(segs, list) and segs:
            snapped = []
            for seg in sorted(segs, key=lambda s: float(s.get("start_time", 0))):
                try:
                    s = _snap_start(words, float(seg["start_time"]), sentence=False)
                    e = _snap_end(words, float(seg["end_time"]), sentence=False)
                except (KeyError, TypeError, ValueError):
                    continue
                s, e = max(s, new_start), min(e, new_end)
                if snapped and s < snapped[-1]["end_time"]:
                    s = snapped[-1]["end_time"]
                if e - s > 0.3:
                    snapped.append({**seg, "start_time": round(s, 3), "end_time": round(e, 3)})
            if snapped:
                snapped[0]["start_time"] = clip["start_time"]
                snapped[-1]["end_time"] = clip["end_time"]
                clip["keep_segments"] = snapped

        if abs(new_start - old_start) > 0.05 or abs(new_end - old_end) > 0.05:
            print(
                f"   ✂️  Rank {clip.get('rank', '?')}: boundaries snapped "
                f"{old_start:.2f}-{old_end:.2f}s → {clip['start_time']:.2f}-{clip['end_time']:.2f}s"
            )
    return clips
