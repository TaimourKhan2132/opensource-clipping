"""
Active-speaker camera planning for vertical crops of multi-person footage.

The default trackers frame the *largest* face. In TV scenes with several people
that picks the wrong person, and when two faces are similar in size the choice
flips back and forth, making the crop jump several times a second.

This planner:
  1. splits the clip into camera shots (frame-difference cut detection),
  2. tracks faces inside each shot with MediaPipe FaceLandmarker,
  3. scores each face's mouth movement ("jawOpen" blendshape) while the
     transcript says someone is speaking,
  4. picks a target per shot with hysteresis: a new speaker must lead for
     SWITCH_AFTER seconds and the current one must have been held MIN_HOLD
     seconds, and
  5. frames the whole group when everyone fits inside the crop.

The result is a list of {"time", "cx", "cy"} points (clip-relative seconds)
in the same format as the other camera planners in render_hybrid.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

SAMPLE_STEP = 0.125      # seconds between face analyses
LANDMARK_CROP = 256      # each face crop is resized to this square for the landmarker
CUT_DIFF = 25.0          # mean abs difference (0-255) on a 64x36 thumbnail needed for a cut...
CUT_SPIKE = 4.0          # ...and it must be this many times the recent motion level, so fast
                         # motion (action scenes, whip pans) is not mistaken for a cut
SPEECH_PAD = 0.15        # seconds around each transcribed word that count as speech
SCORE_WINDOW = 0.5       # +/- seconds of mouth movement averaged into the speaking score
MIN_SCORE = 0.015        # below this nobody is clearly talking; keep the current target
SWITCH_AFTER = 0.5       # a challenger must lead this long before the camera switches
MIN_HOLD = 1.2           # never leave a target sooner than this after switching to it
LEAD_MARGIN = 1.3        # challenger score must beat the current target by this factor
GROUP_FILL = 0.85        # frame the group if all faces span at most this share of the crop
SMOOTH_WINDOW = 0.4      # +/- seconds of centered smoothing while following one target
EASE_TIME = 0.3          # short pans between nearby targets; far targets get a hard cut
HARD_CUT_SHARE = 0.35    # target moves larger than this share of crop width are hard cuts
HEAD_FACTOR = 1.8        # window width needed per face-box width to keep the whole head in frame
GROUP_MARGIN = 0.35      # extra face-widths of margin on each side of a framed group
MAX_ZOOM_OUT = 2.0       # widest zoom-out (window = 2x the 9:16 crop; video band = half the height)
MIN_ZOOM_OUT = 1.08      # smaller zoom-outs are skipped (full-screen crop instead)
LOST_HOLD = 3.0          # seconds to hold still after losing the target when nobody is visibly talking
LOOKAHEAD = 1.5          # seconds scanned at a shot start to find who talks first
CONTINUE_SHARE = 0.45    # a face this close to the camera position continues a lost target
MIN_FACE_SHARE = 0.4     # faces narrower than this share of the biggest face are ignored


def _landmarker(model_path):
    if not os.path.exists(model_path):
        print("   📥 Downloading MediaPipe FaceLandmarker model...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _faces(landmarker, detector, frame, width, height):
    """
    Two-stage face analysis. FaceLandmarker's own detector is short-range (selfie
    distance) and misses the smaller faces of medium/wide TV shots, so faces are
    found with the full-range detector, then each face is cropped, enlarged and
    passed to the landmarker to read its jawOpen blendshape.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    det = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = []
    for d in det.detections or []:
        b = d.bounding_box
        if b.width < 12 or b.height < 12:
            continue
        cx, cy = b.origin_x + b.width / 2, b.origin_y + b.height / 2
        side = int(max(b.width, b.height) * 1.9)
        ax, ay = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        crop = rgb[ay: ay + side, ax: ax + side]
        jaw = float("nan")
        if crop.size:
            crop = np.ascontiguousarray(cv2.resize(crop, (LANDMARK_CROP, LANDMARK_CROP)))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if res.face_landmarks and res.face_blendshapes:
                k = min(range(len(res.face_landmarks)),
                        key=lambda i: abs(np.mean([p.x for p in res.face_landmarks[i]]) - 0.5))
                jaw = next((c.score for c in res.face_blendshapes[k] if c.category_name == "jawOpen"), float("nan"))
        sx, sy = width / W, height / H
        faces.append({
            "cx": cx * sx, "cy": cy * sy, "x1": b.origin_x * sx, "x2": (b.origin_x + b.width) * sx,
            "w": b.width * sx, "jaw": jaw,
        })
    return faces


def _speech_mask(times, speech_segments, start_clip):
    intervals = []
    for seg in speech_segments or []:
        for wd in seg.get("words", []) or [seg]:
            try:
                intervals.append((float(wd["start"]) - start_clip - SPEECH_PAD, float(wd["end"]) - start_clip + SPEECH_PAD))
            except (KeyError, TypeError, ValueError):
                continue
    if not intervals:
        return np.ones(len(times), bool)  # no transcript: trust mouth movement alone
    intervals.sort()
    mask = np.zeros(len(times), bool)
    for i, t in enumerate(times):
        mask[i] = any(a <= t <= b for a, b in intervals if a <= t + 1 and b >= t - 1)
    return mask


def _analyse(input_video, start_clip, end_clip, width, height, model_path, detector, label):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
    step_frames = max(1, int(round(SAMPLE_STEP * fps)))
    duration = end_clip - start_clip
    landmarker = _landmarker(model_path)

    samples, cut_times = [], [0.0]
    prev_thumb, idx, force_sample, last_pct, recent = None, 0, True, -1, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / fps
        if t > duration:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        thumb = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_thumb is not None:
            diff = float(np.mean(np.abs(thumb - prev_thumb)))
            base = float(np.median(recent[-6:])) if recent else 0.0
            if diff > CUT_DIFF and diff / (base + 3.0) > CUT_SPIKE:
                cut_times.append(t)
                force_sample = True
            recent.append(diff)
        prev_thumb = thumb
        if force_sample or idx % step_frames == 0:
            samples.append({"t": t, "shot": len(cut_times) - 1, "faces": _faces(landmarker, detector, frame, width, height)})
            force_sample = False
        pct = int(100 * t / duration) if duration > 0 else 100
        if pct // 10 != last_pct // 10:
            print(f"⏳ {label} - Speaker analysis: {pct:3d}%", flush=True)
            last_pct = pct
        idx += 1
    cap.release()
    landmarker.close()
    return samples, cut_times


def _track(samples):
    """Give faces a track id that is stable within a shot (nearest-center matching)."""
    next_id, last_in_shot, shot = 0, [], None
    for s in samples:
        if s["shot"] != shot:
            shot, last_in_shot = s["shot"], []
        used = set()
        for f in sorted(s["faces"], key=lambda f: -f["w"]):
            best, best_d = None, None
            for tr in last_in_shot:
                if tr["id"] in used:
                    continue
                d = abs(tr["cx"] - f["cx"]) + abs(tr["cy"] - f["cy"])
                ratio = f["w"] / max(tr["w"], 1)
                if d < 1.2 * max(tr["w"], f["w"]) and 0.5 < ratio < 2.0 and (best_d is None or d < best_d):
                    best, best_d = tr, d
            if best is None:
                f["id"] = next_id
                next_id += 1
            else:
                f["id"] = best["id"]
            used.add(f["id"])
        seen = {f["id"] for f in s["faces"]}
        last_in_shot = [f for f in s["faces"]] + [tr for tr in last_in_shot if tr["id"] not in seen]


def _scores(samples, speaking):
    """Mouth-movement score per (sample index, track id), zero when nobody is talking."""
    last_jaw, motion = {}, []
    for s in samples:
        m = {}
        for f in s["faces"]:
            key = (s["shot"], f["id"])
            if np.isnan(f["jaw"]):
                continue  # face found but mouth unreadable (profile, blur)
            if key in last_jaw:
                m[f["id"]] = abs(f["jaw"] - last_jaw[key])
            last_jaw[key] = f["jaw"]
        motion.append(m)
    times = np.array([s["t"] for s in samples])
    scores = []
    for i, s in enumerate(samples):
        if not speaking[i]:
            scores.append({})
            continue
        lo, hi = np.searchsorted(times, s["t"] - SCORE_WINDOW), np.searchsorted(times, s["t"] + SCORE_WINDOW, "right")
        sc = {}
        for f in s["faces"]:
            vals = [motion[j][f["id"]] for j in range(lo, hi) if samples[j]["shot"] == s["shot"] and f["id"] in motion[j]]
            sc[f["id"]] = float(np.mean(vals)) if vals else 0.0
        scores.append(sc)
    return scores


def _choose_targets(samples, scores, crop_w, width):
    """
    Per sample: (shot, target_key, x). target_key only changes on a real switch
    (new speaker, group framing, or re-pick after the target is gone), so losing
    and re-finding the same face does not move the camera.
    """
    out, shot = [], None
    cur = cur_since = challenger = challenger_since = cur_seen = None
    key, cam_x, last_x = 0, width / 2, {}

    def first_speaker(i, faces):
        # Offline advantage: at a shot start, look ahead for who talks first
        acc = {}
        for j in range(i, len(samples)):
            if samples[j]["shot"] != samples[i]["shot"] or samples[j]["t"] - samples[i]["t"] > LOOKAHEAD:
                break
            for fid, v in scores[j].items():
                acc.setdefault(fid, []).append(v)
        ids = {f["id"] for f in faces}
        means = {fid: float(np.mean(v)) for fid, v in acc.items() if fid in ids}
        if means and max(means.values()) >= MIN_SCORE / 2:
            return max(means, key=means.get)
        return max(faces, key=lambda f: f["w"])["id"]

    for i, (s, sc) in enumerate(zip(samples, scores)):
        t = s["t"]
        faces = s["faces"]
        if faces:  # ignore tiny faces (reflections, background) when a bigger one is present
            biggest = max(f["w"] for f in faces)
            faces = [f for f in faces if f["w"] >= MIN_FACE_SHARE * biggest]
        if s["shot"] != shot:
            shot, cur, challenger, last_x = s["shot"], None, None, {}
            key += 1
        by_id = {f["id"]: f for f in faces}
        desired = None
        if len(faces) >= 2 and max(f["x2"] for f in faces) - min(f["x1"] for f in faces) <= GROUP_FILL * crop_w:
            desired = "group"
        else:
            visible = {i: v for i, v in sc.items() if i in by_id}
            if visible:
                best = max(visible, key=visible.get)
                if visible[best] >= MIN_SCORE:
                    desired = best
        if (cur == "group" and faces) or cur in by_id:
            cur_seen = t

        if faces and cur is None:
            cur = desired if desired is not None else first_speaker(i, faces)
            cur_since, cur_seen, challenger = t, t, None
        elif faces and cur != "group" and cur not in by_id:
            near = min(faces, key=lambda f: abs(f["cx"] - cam_x))
            if abs(near["cx"] - cam_x) <= CONTINUE_SHARE * crop_w:
                cur, cur_seen = near["id"], t  # same person re-detected: keep following, no switch
            elif (desired is not None and t - cur_seen > 0.6) or t - cur_seen > LOST_HOLD:
                # Move only for a visible speaker; with no evidence, hold still a while
                cur = desired if desired is not None else max(faces, key=lambda f: f["w"])["id"]
                cur_since, cur_seen, challenger = t, t, None
                key += 1
        elif desired is not None and desired != cur:
            cur_score = sc.get(cur, 0.0) if cur != "group" else 0.0
            leads = desired == "group" or sc.get(desired, 0.0) > LEAD_MARGIN * cur_score
            if challenger != desired or not leads:
                challenger, challenger_since = (desired, t) if leads else (None, None)
            elif t - challenger_since >= SWITCH_AFTER and t - cur_since >= MIN_HOLD:
                cur, cur_since, cur_seen, challenger = desired, t, t, None
                key += 1
        else:
            challenger = None

        need = None  # window width that keeps the framed head(s) whole
        if cur == "group" and faces:
            x = (max(f["x2"] for f in faces) + min(f["x1"] for f in faces)) / 2
            need = (max(f["x2"] for f in faces) - min(f["x1"] for f in faces)
                    + 2 * GROUP_MARGIN * float(np.mean([f["w"] for f in faces])))
        elif cur in by_id:
            x = by_id[cur]["cx"]
            need = by_id[cur]["w"] * HEAD_FACTOR
        elif cur in last_x:
            x = last_x[cur]
        else:
            x = cam_x if out and out[-1][0] == shot else width / 2
        last_x[cur] = cam_x = x
        out.append((shot, key, x, need))
    return out


def plan_speaker_camera(input_video, start_clip, end_clip, width, height, crop_w, speech_segments, model_path, detector, label="Speaker"):
    """Return camera points [{"time", "cx", "cy"}] for a vertical crop that follows the active speaker."""
    print(f"🧠 {label} - Speaker analysis started (shots, faces, mouth movement)...", flush=True)
    samples, cut_times = _analyse(input_video, start_clip, end_clip, width, height, model_path, detector, label)
    if not samples:
        return []
    _track(samples)
    speaking = _speech_mask([s["t"] for s in samples], speech_segments, start_clip)
    chosen = _choose_targets(samples, _scores(samples, speaking), crop_w, width)

    # Smooth within runs of (shot, target); hard cut or short ease between runs
    times = np.array([s["t"] for s in samples])
    xs = np.array([c[2] for c in chosen], dtype=float)

    # One zoom level per camera shot (changes only at cuts, so it never pumps)
    zoom_by_shot = {}
    for shot in {c[0] for c in chosen}:
        needs = [c[3] / crop_w for c in chosen if c[0] == shot and c[3]]
        z = float(np.percentile(needs, 75)) if needs else 1.0
        zoom_by_shot[shot] = 1.0 if z < MIN_ZOOM_OUT else min(z, MAX_ZOOM_OUT, width / crop_w)
    runs, start = [], 0
    for i in range(1, len(chosen) + 1):
        if i == len(chosen) or chosen[i][:2] != chosen[i - 1][:2]:
            runs.append((start, i))
            start = i
    k = max(1, int(round(SMOOTH_WINDOW / SAMPLE_STEP)))
    smooth = xs.copy()
    for a, b in runs:
        for i in range(a, b):
            smooth[i] = xs[max(a, i - k): min(b, i + k + 1)].mean()

    cy = height / 2
    points = []
    for n, (a, b) in enumerate(runs):
        if n > 0:
            prev_x = points[-1]["cx"]
            new_shot = chosen[a][0] != chosen[a - 1][0]
            if new_shot or abs(smooth[a] - prev_x) > HARD_CUT_SHARE * crop_w:
                points.append({"time": max(times[a] - 1e-3, points[-1]["time"] + 1e-4), "cx": prev_x, "cy": cy,
                               "zoom": points[-1]["zoom"]})
            else:
                for i in range(a, b):
                    u = (times[i] - times[a]) / EASE_TIME
                    if u >= 1:
                        break
                    e = u * u * (3 - 2 * u)
                    smooth[i] = prev_x + (smooth[i] - prev_x) * e
        for i in range(a, b):
            points.append({"time": float(times[i]), "cx": float(smooth[i]), "cy": cy, "zoom": zoom_by_shot[chosen[i][0]]})

    n_switch = sum(1 for n in range(1, len(runs)) if chosen[runs[n][0]][0] == chosen[runs[n - 1][0]][0])
    n_zoom = sum(1 for z in zoom_by_shot.values() if z > 1.0)
    print(f"✅ {label} - {len(cut_times)} shots, {n_switch} in-shot speaker switches, "
          f"{n_zoom} shots zoomed out to keep heads in frame.", flush=True)
    return points
