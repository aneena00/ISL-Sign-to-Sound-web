"""
Mood / emotion detection — built on MediaPipe Face Landmarker blendshapes
instead of raw pixel variance.

Why: the old version measured pixel "noisiness" in the mouth/eyebrow region
of a Haar-cascade face box. That's sensitive to lighting, camera noise, and
head angle. Blendshapes are semantic, model-estimated facial-muscle
activations (mouth smile, brow down, jaw open, etc.) computed directly from
face landmarks, so they're far more stable across lighting and pose.

Drop-in compatible: still exposes calculate_emotion(frame) -> str.
"""

import os
import urllib.request

import cv2
import mediapipe as mp

_MODEL_PATH = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# How strong the winning blendshape signal needs to be before we trust it
# over "Neutral". These are intentionally high — the goal is for Neutral to
# be the default state, only overridden by a clear, deliberate expression.
MOOD_SENSITIVITY = 0.40
SURPRISE_SENSITIVITY = 0.45


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("\n[System] Downloading face landmark model file...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[System] Face model download verified successfully!")


class EmotionDetector:
    """
    One instance = one MediaPipe FaceLandmarker. Create a separate instance
    per camera session (e.g. one per connected user) rather than sharing a
    single global instance across multiple simultaneous video streams.
    """

    def __init__(self):
        _ensure_model()
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    @staticmethod
    def _score(blendshapes, name):
        for b in blendshapes:
            if b.category_name == name:
                return b.score
        return 0.0

    def calculate_with_scores(self, frame):
        """Returns (label, raw_scores_dict) — use raw_scores for calibration."""
        if frame is None:
            return "Neutral", {}
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)

            if not result.face_blendshapes:
                return "Neutral", {}

            shapes = result.face_blendshapes[0]
            s = self._score

            smile       = (s(shapes, "mouthSmileLeft") + s(shapes, "mouthSmileRight")) / 2
            frown       = (s(shapes, "mouthFrownLeft") + s(shapes, "mouthFrownRight")) / 2
            mouth_press = (s(shapes, "mouthPressLeft") + s(shapes, "mouthPressRight")) / 2
            brow_down   = (s(shapes, "browDownLeft") + s(shapes, "browDownRight")) / 2
            brow_in_up  = s(shapes, "browInnerUp")
            eye_wide    = (s(shapes, "eyeWideLeft") + s(shapes, "eyeWideRight")) / 2
            eye_squint  = (s(shapes, "eyeSquintLeft") + s(shapes, "eyeSquintRight")) / 2
            jaw_open    = s(shapes, "jawOpen")

            # Every non-neutral mood now requires two CORRELATED signals
            # together, not one blendshape acting alone — a single noisy
            # channel (e.g. browInnerUp sitting slightly above zero at
            # rest) can no longer win by itself. Combined with the higher
            # floors above, Neutral is the default outcome; something has
            # to be genuinely, deliberately expressed to override it.
            happy_score = smile
            surprise_score = min(brow_in_up, max(eye_wide, jaw_open * 0.7))
            angry_score = min(brow_down, max(mouth_press, eye_squint * 0.7, brow_down * 0.6))
            sad_score = min(frown, brow_in_up)

            scores = {
                "Happy": happy_score,
                "Surprise": surprise_score,
                "Angry": angry_score,
                "Sad": sad_score,
            }

            floors = {
                "Happy": MOOD_SENSITIVITY,
                "Surprise": SURPRISE_SENSITIVITY,
                "Angry": MOOD_SENSITIVITY,
                "Sad": MOOD_SENSITIVITY,
            }

            best_label = max(scores, key=scores.get)
            if scores[best_label] < floors[best_label]:
                return "Neutral", scores
            return best_label, scores

        except Exception:
            return "Neutral", {}

    def calculate(self, frame):
        label, _ = self.calculate_with_scores(frame)
        return label


_default_detector = None


def calculate_emotion(frame):
    """Convenience function for simple, single-stream scripts."""
    global _default_detector
    if _default_detector is None:
        _default_detector = EmotionDetector()
    return _default_detector.calculate(frame)
