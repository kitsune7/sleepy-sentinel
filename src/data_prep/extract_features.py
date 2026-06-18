#!/usr/bin/env python3
"""
extract_features.py
====================

Stream per-frame facial features from the drowsiness-dataset videos using
MediaPipe FaceLandmarker and write ONE small CSV per video. Frames are read and
processed one at a time, so a multi-hundred-MB video never lives in RAM, and you
can point this at a single file (process, then delete the source) so the full
~111 GB dataset never needs to sit on disk at once.

Dataset layout expected (one folder per subject, three videos per folder):

    <root>/
        01/  0.mov   5.mov   10.mov
        02/  0.mov   5.mov   10.mov
        ...

Filename -> ordinal label:

    0.mov  -> 0  (alert)
    5.mov  -> 1  (low-vigilant)
    10.mov -> 2  (drowsy)

Output (tiny text files, one row per frame):

    <out>/
        01/  0.csv   5.csv   10.csv
        ...

Dependencies
------------
    uv add mediapipe opencv-python numpy

Examples
--------
    # single video (good for the "process then delete" loop)
    uv run extract_features --video data/01/0.mov --subject 01 --label 0 --out features

    # whole tree, resume-safe, keep going past per-video errors
    uv run extract_features --root data --out features

    # reclaim space as you go (deletes each source AFTER its CSV is written)
    uv run extract_features --root data --out features --delete-source
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# filename stem -> ordinal class label
LABEL_MAP = {"0": 0, "5": 1, "10": 2}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Conventional MediaPipe FaceMesh 6-point eye landmark indices for EAR.
# Order per eye: [h_corner_a, v_top_a, v_top_b, h_corner_b, v_bot_b, v_bot_a].
# These are the widely-used indices; verify once by overlaying them on a frame.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Blendshape category names emitted by FaceLandmarker.
BS_BLINK_L = "eyeBlinkLeft"
BS_BLINK_R = "eyeBlinkRight"
BS_JAW_OPEN = "jawOpen"

# CSV columns written per frame.
COLUMNS = [
    "frame_idx", "t_ms", "face",
    "eye_blink_l", "eye_blink_r", "ear",
    "jaw_open",
    "pitch", "yaw", "roll",
    "bright_mean", "warmth",   # for the luminance-confound baseline
]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def eye_aspect_ratio(landmarks, idx, w, h):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||) on pixel coords."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx])
    vert = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-6:
        return float("nan")
    return float(vert / (2.0 * horiz))


def rotation_to_euler(matrix):
    """Decompose a 4x4 facial transformation matrix into (pitch, yaw, roll) deg.

    Axis naming is approximate; the model only needs *consistent* angles (we use
    their variance/range), so an axis swap or sign flip won't hurt training.
    Sanity-check the labels visually if you ever use a raw angle as a feature.
    """
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


def frame_photometrics(frame_bgr):
    """Cheap whole-frame brightness and warmth (R/B) for the lighting baseline."""
    b = float(frame_bgr[..., 0].mean())
    g = float(frame_bgr[..., 1].mean())
    r = float(frame_bgr[..., 2].mean())
    bright = (b + g + r) / 3.0
    warmth = (r + 1.0) / (b + 1.0)
    return bright, warmth


def blendshape_dict(face_blendshapes):
    """Map the first face's blendshape categories to {name: score}."""
    if not face_blendshapes:
        return {}
    return {c.category_name: c.score for c in face_blendshapes[0]}


def download_model(model_path, url=MODEL_URL):
    """Download the FaceLandmarker task bundle atomically."""
    target = os.path.abspath(model_path)
    tmp_path = f"{target}.download"
    os.makedirs(os.path.dirname(target), exist_ok=True)

    print(f"[dl]   {url} -> {target}")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def format_duration(seconds):
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:d}:{secs:02d}"


def render_progress(label, current, total, started_at, force=False):
    now = time.monotonic()
    if not force and now - render_progress.last_update < 0.25:
        return
    render_progress.last_update = now

    elapsed = max(now - started_at, 1e-6)
    fps = current / elapsed
    if total > 0:
        pct = min(current / total, 1.0)
        width = 30
        filled = int(width * pct)
        bar = "#" * filled + "-" * (width - filled)
        remaining = max(total - current, 0) / fps if fps > 0 else 0
        line = (
            f"\r[prog] {label} [{bar}] {pct * 100:5.1f}% "
            f"{current}/{total} frames, {fps:5.1f} fps, eta {format_duration(remaining)}"
        )
    else:
        line = f"\r[prog] {label} {current} frames, {fps:5.1f} fps"
    sys.stderr.write(line)
    sys.stderr.flush()


render_progress.last_update = 0.0


# --------------------------------------------------------------------------- #
# Core: one video -> one CSV (streaming, frame by frame)
# --------------------------------------------------------------------------- #

def make_landmarker(model_path):
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def process_video(video_path, out_csv, landmarker, frame_stride=1, show_progress=True):
    """Read `video_path` frame by frame, write per-frame features to `out_csv`.

    Returns (n_frames_written, n_faces_detected). RAM stays bounded by a single
    decoded frame plus a tiny row buffer (~a few MB even for long clips).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_label = os.path.join(os.path.basename(os.path.dirname(video_path)), os.path.basename(video_path))
    progress_started_at = time.monotonic()
    render_progress.last_update = 0.0
    rows = []
    frame_idx = -1
    n_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if show_progress:
            render_progress(progress_label, frame_idx + 1, total_frames, progress_started_at)
        if frame_stride > 1 and (frame_idx % frame_stride):
            continue

        h, w = frame.shape[:2]
        bright, warmth = frame_photometrics(frame)

        # MediaPipe expects RGB; timestamps must increase monotonically.
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not t_ms or t_ms <= 0:
            t_ms = (frame_idx / fps) * 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(t_ms))

        if result.face_landmarks:
            n_faces += 1
            lms = result.face_landmarks[0]
            bs = blendshape_dict(result.face_blendshapes)
            ear_l = eye_aspect_ratio(lms, LEFT_EYE_IDX, w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE_IDX, w, h)
            ear = np.nanmean([ear_l, ear_r])
            if result.facial_transformation_matrixes:
                pitch, yaw, roll = rotation_to_euler(
                    result.facial_transformation_matrixes[0]
                )
            else:
                pitch = yaw = roll = float("nan")
            rows.append([
                frame_idx, round(t_ms, 2), 1,
                bs.get(BS_BLINK_L, float("nan")),
                bs.get(BS_BLINK_R, float("nan")),
                round(float(ear), 5),
                bs.get(BS_JAW_OPEN, float("nan")),
                round(pitch, 3), round(yaw, 3), round(roll, 3),
                round(bright, 3), round(warmth, 5),
            ])
        else:
            # No face this frame: flag it, leave feature columns as NaN.
            nan = float("nan")
            rows.append([
                frame_idx, round(t_ms, 2), 0,
                nan, nan, nan, nan, nan, nan, nan,
                round(bright, 3), round(warmth, 5),
            ])

    cap.release()
    if show_progress:
        render_progress(progress_label, frame_idx + 1, total_frames, progress_started_at, force=True)
        sys.stderr.write("\n")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    return len(rows), n_faces


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

def iter_videos(root):
    """Yield (subject_id, label, video_path) for every N/{0,5,10}.mov under root."""
    for subject in sorted(os.listdir(root)):
        sub_dir = os.path.join(root, subject)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() == ".mov" and stem in LABEL_MAP:
                yield subject, LABEL_MAP[stem], os.path.join(sub_dir, fname)


def out_path_for(out_root, subject, video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_root, subject, f"{stem}.csv")


def run_one(video_path, out_csv, landmarker, args):
    if os.path.exists(out_csv) and not args.overwrite:
        print(f"[skip] {out_csv} exists")
        return
    n, faces = process_video(video_path, out_csv, landmarker, args.frame_stride, not args.no_progress)
    pct = (100.0 * faces / n) if n else 0.0
    print(f"[ok]   {video_path} -> {out_csv}  ({n} frames, {pct:.1f}% with a face)")
    if args.delete_source:
        os.remove(video_path)
        print(f"[rm]   {video_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="face_landmarker.task",
                    help="path to the FaceLandmarker .task bundle")
    ap.add_argument("--model-url", default=MODEL_URL,
                    help="URL to download the FaceLandmarker .task bundle from")
    ap.add_argument("--out", help="output root for the CSVs")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="process every Nth frame (1 = all; keep 1 for blink fidelity)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even if the output CSV already exists")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete each source video AFTER its CSV is written")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable per-video terminal progress bars")
    ap.add_argument("--keep-going", action="store_true", default=True,
                    help="continue past per-video errors (default on)")
    # one of:
    ap.add_argument("--root", help="dataset root containing subject folders")
    ap.add_argument("--video", help="single video file (use with --subject/--label)")
    ap.add_argument("--subject", help="subject id for --video mode")
    ap.add_argument("--label", help="filename stem (0/5/10) for --video mode")
    args = ap.parse_args()

    if os.path.exists(args.model):
        print(f"[skip] {args.model} exists")
    else:
        download_model(args.model, args.model_url)
    if not args.video and not args.root:
        return

    if not args.video and not args.root:
        sys.exit("provide either --root or --video")
    if not args.out:
        sys.exit("provide --out for extraction output")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}\n"
                 f"download it with:\n  curl -L \"{args.model_url}\" -o {args.model}\n"
                 "or rerun with --download-model.")

    print("Starting feature extraction...")
    landmarker = make_landmarker(args.model)
    try:
        if args.video:
            subject = args.subject or "unknown"
            out_csv = out_path_for(args.out, subject, args.video)
            run_one(args.video, out_csv, landmarker, args)
        elif args.root:
            for subject, _label, vpath in iter_videos(args.root):
                out_csv = out_path_for(args.out, subject, vpath)
                try:
                    run_one(vpath, out_csv, landmarker, args)
                except Exception:
                    if not args.keep_going:
                        raise
                    print(f"[err]  {vpath}\n{traceback.format_exc()}")
    finally:
        landmarker.close()
        print("Feature extraction completed successfully.")


if __name__ == "__main__":
    main()
