import os
import io
import json
import time
import threading
import asyncio  # <-- ADDED FOR THREAD PATCH
from datetime import datetime

import av
import cv2
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- CRITICAL STREAMLIT CLOUD LIFECYCLE PATCH ---
# This prevents asyncio from tearing down event loops while webrtc sockets are active
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from calculate_emotion import EmotionDetector
except ImportError:
    EmotionDetector = None

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH = "isl_model.pth"
REF_IMG_DIR = ""  
LABELS_FILE = "labels.json"
DEFAULT_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  

SEQ_LEN = 30
CONFIDENCE_THRESHOLD = 0.80
STABILITY_COUNT = 3
DEBOUNCE_DELAY = 1.5

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# HARDENED WEBRTC TRAVERSAL SHIELD
# Bypasses the broken openrelay project with globally redundant Google and Twilio public endpoints
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {
                "urls": ["turn:global.turn.twilio.com:3478?transport=udp", "turn:global.turn.twilio.com:443?transport=tcp"],
                "username": "2b09be8b082fe286e969d67ba5b4c1ea5dfa04cdd444c12513ba6b2ca8859942",
                "credential": "NzU1MmNmYmU2MzhhMmIzYjFmN2NiNzg5MDlhMTc0N2EwNDBhMTYwMzA4MDNkMDRjZjM0YmFmODdiZWY0OTVlYw==",
            }
        ]
    }
)

# --------------------------------------------------------------------------
# PAGE + THEME
# --------------------------------------------------------------------------
st.set_page_config(page_title="Sign2Sound", layout="wide", page_icon="🤟")

INK = "#0B0E14"
PANEL = "#141923"
MARIGOLD = "#FFB627"
TEAL = "#2EC4B6"
MAGENTA = "#E6539A"
MUTED = "#8B93A7"

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

        .stApp {{ background-color: {INK}; color: #EDEFF4; }}
        h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}
        p, span, label, div {{ font-family: 'Inter', sans-serif; }}

        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 44px; font-weight: 700; margin-bottom: 0;
            background: linear-gradient(90deg, {TEAL}, {MARIGOLD}, {MAGENTA});
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }}
        .hero-sub {{ color: {MUTED}; font-size: 16px; margin-top: 0; margin-bottom: 18px; }}
        .bridge-line {{
            height: 4px; border-radius: 4px; margin-bottom: 28px;
            background: linear-gradient(90deg, {TEAL}, {MARIGOLD}, {MAGENTA});
            background-size: 200% 100%;
            animation: shimmer 6s ease-in-out infinite;
        }}
        @keyframes shimmer {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .panel {{
            background: {PANEL}; border-radius: 14px; padding: 18px 20px;
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .mono-status {{
            font-family: 'JetBrains Mono', monospace; font-size: 20px;
            font-weight: 500; margin: 6px 0;
        }}
        .status-teal {{ color: {TEAL}; }}
        .status-marigold {{ color: {MARIGOLD}; }}
        .status-muted {{ color: {MUTED}; }}
        .word-pill {{
            display: inline-block; font-family: 'JetBrains Mono', monospace;
            background: rgba(255,182,39,0.12); color: {MARIGOLD};
            border: 1px solid rgba(255,182,39,0.4); border-radius: 8px;
            padding: 4px 14px; font-size: 22px; font-weight: 500;
            white-space: pre;
        }}

        div[data-testid="stButton"] button {{
            background: linear-gradient(90deg, {TEAL}, {MARIGOLD});
            color: {INK}; border: none; border-radius: 8px; font-weight: 600;
            font-family: 'Inter', sans-serif;
        }}
        div[data-testid="stButton"] button:hover {{ filter: brightness(1.08); color: {INK}; }}

        .stTextInput input {{
            background-color: {PANEL} !important; color: #EDEFF4 !important;
            border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 8px !important;
        }}

        .chat-scroll {{
            max-height: 320px; overflow-y: auto; padding: 6px 4px;
            display: flex; flex-direction: column; gap: 10px;
        }}
        .bubble {{
            max-width: 70%; padding: 10px 14px; border-radius: 14px;
            font-size: 15px; line-height: 1.4;
        }}
        .bubble-you {{
            align-self: flex-end; background: rgba(46,196,182,0.18);
            border: 1px solid {TEAL}; color: #DFFFFB;
        }}
        .bubble-friend {{
            align-self: flex-start; background: rgba(230,83,154,0.18);
            border: 1px solid {MAGENTA}; color: #FFE3F2;
        }}
        .bubble-meta {{ font-size: 11px; color: {MUTED}; margin-top: 3px; }}
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
defaults = {
    "received_text": "",
    "chat_log": [],   
    "last_audio_bytes": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------------------------------
# CACHED, READ-ONLY RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource
def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            return json.load(f)
    return DEFAULT_LABELS


@st.cache_resource
def load_model(num_classes):
    m = SignLSTM(num_classes=num_classes)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        st.error(f"Model weights not found at '{MODEL_PATH}'. Train the model first.")
    m.eval()
    return m


LABELS = load_labels()
MODEL = load_model(len(LABELS))


# --------------------------------------------------------------------------
# BROWSER-SIDE SPEECH
# --------------------------------------------------------------------------
def speak_in_browser(text):
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    components.html(
        f"""<script>
            try {{
                var msg = new SpeechSynthesisUtterance("{safe}");
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(msg);
            }} catch (e) {{}}
        </script>""",
        height=0, width=0,
    )


# --------------------------------------------------------------------------
# PER-CONNECTION VIDEO PROCESSOR
# --------------------------------------------------------------------------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.extractor = MediaPipeFeatureExtractor()

        self.emotion_detector = None
        if EmotionDetector is not None:
            try:
                self.emotion_detector = EmotionDetector()
            except Exception as e:
                print(f"[Warning] Mood detection disabled — could not load face model: {e}")

        self.sequence = []
        self.stability_buffer = []
        self.current_word = ""
        self.last_added_letter = ""
        self.last_addition_time = 0.0

        self.sign_text = "Searching for hands..."
        self.mood = "Neutral"
        self.mood_scores = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        mood_scores = {}
        if self.emotion_detector is not None:
            try:
                mood, mood_scores = self.emotion_detector.calculate_with_scores(img)
            except Exception:
                mood = "Neutral"
        else:
            mood = "Unavailable"

        features = self.extractor.extract_features(img)

        canvas = np.zeros_like(img)
        if self.extractor.hands_result and self.extractor.hands_result.hand_landmarks:
            for hand_landmarks in self.extractor.hands_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for x, y in pts:
                    cv2.circle(canvas, (x, y), 5, (60, 200, 255), -1)
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
                for a, b in HAND_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(canvas, pts[a], pts[b], (200, 120, 255), 2)

        with self.lock:
            if np.any(features):
                self.sequence.append(features)
                self.sequence = self.sequence[-SEQ_LEN:]

                if len(self.sequence) == SEQ_LEN:
                    with torch.no_grad():
                        inp = torch.tensor(np.array([self.sequence]), dtype=torch.float32)
                        out = MODEL(inp)
                        prob = torch.nn.functional.softmax(out, dim=1)
                        conf, idx = torch.max(prob, dim=1)
                        prediction = LABELS[idx.item()]
                        confidence = conf.item()

                    self.stability_buffer.append(prediction)
                    self.stability_buffer = self.stability_buffer[-5:]

                    sign_text = f"Detecting... ({prediction}, {confidence:.2f})"
                    if (confidence >= CONFIDENCE_THRESHOLD
                            and self.stability_buffer.count(prediction) >= STABILITY_COUNT):
                        sign_text = f"Confirmed: {prediction} ({confidence:.2f})"
                        now = time.time()
                        if (prediction != self.last_added_letter
                                or (now - self.last_addition_time > DEBOUNCE_DELAY)):
                            self.current_word += prediction
                            self.last_added_letter = prediction
                            self.last_addition_time = now
                else:
                    sign_text = f"Buffering... ({len(self.sequence)}/{SEQ_LEN})"
            else:
                self.sequence = []
                self.stability_buffer = []
                self.last_added_letter = ""
                sign_text = "Searching for hands..."

            self.sign_text = sign_text
            self.mood = mood
            self.mood_scores = mood_scores

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")


# --------------------------------------------------------------------------
# UI RENDER
# --------------------------------------------------------------------------
st.markdown("<div class='hero-title'>Sign2Sound</div>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>A live bridge between ISL signs and spoken word.</p>", unsafe_allow_html=True)
st.markdown("<div class='bridge-line'></div>", unsafe_allow_html=True)

col_cam, col_reply = st.columns(2)

with col_cam:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("##### Your Camera")

    ctx = webrtc_streamer(
        key="sign2sound",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )

    sign_placeholder = st.empty()
    word_placeholder = st.empty()
    mood_placeholder = st.empty()
    with st.expander("Mood debug scores"):
        mood_debug_placeholder = st.empty()

    bspeak, bspace, bback, bclear = st.columns(4)
    speak_clicked = bspeak.button("Speak", width="stretch")
    space_clicked = bspace.button("␣ Space", width="stretch")
    back_clicked = bback.button("⌫ Back", width="stretch")
    clear_clicked = bclear.button("Clear", width="stretch")

    if ctx.video_processor:
        if speak_clicked:
            with ctx.video_processor.lock:
                word = ctx.video_processor.current_word
                ctx.video_processor.current_word = ""
                ctx.video_processor.last_added_letter = ""
            if word.strip():
                speak_in_browser(word)
                st.session_state.chat_log.append(
                    {"who": "you", "text": word, "time": datetime.now().strftime("%H:%M")}
                )
        if space_clicked:
            with ctx.video_processor.lock:
                ctx.video_processor.current_word += " "
                ctx.video_processor.last_added_letter = ""
        if back_clicked:
            with ctx.video_processor.lock:
                ctx.video_processor.current_word = ctx.video_processor.current_word[:-1]
                ctx.video_processor.last_addition_time = time.time()
        if clear_clicked:
            with ctx.video_processor.lock:
                ctx.video_processor.current_word = ""
                ctx.video_processor.last_added_letter = ""

    st.markdown("</div>", unsafe_allow_html=True)

with col_reply:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("##### Reply (Two-Way Channel)")

    typed_reply = st.text_input("Type a reply to speak back:", key="typed_reply")
    if st.button("Send Reply") and typed_reply.strip():
        st.session_state.received_text = typed_reply.strip()
        speak_in_browser(typed_reply.strip())
        st.session_state.chat_log.append(
            {"who": "friend", "text": typed_reply.strip(), "time": datetime.now().strftime("%H:%M")}
        )

    st.markdown(
        <p class='status-muted' style='margin:10px 0 4px;'>— or speak it —</p>,
        unsafe_allow_html=True,
    )

    if audio_recorder is not None and sr is not None:
        audio_bytes = audio_recorder(
            text="Click, speak, click again",
            recording_color=MAGENTA,
            neutral_color=TEAL,
            icon_size="2x",
        )
        if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
            st.session_state.last_audio_bytes = audio_bytes
            with st.spinner("Transcribing..."):
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                    heard = recognizer.recognize_google(audio_data)
                except Exception as e:
                    heard = None
                    st.warning("Didn't catch that clearly — speak closer to the microphone.")

            if heard:
                st.session_state.received_text = heard
                speak_in_browser(heard)
                st.session_state.chat_log.append(
                    {"who": "friend", "text": heard, "time": datetime.now().strftime("%H:%M")}
                )

    st.markdown("###### Now Translating")
    if st.session_state.received_text:
        letters = [c for c in st.session_state.received_text.upper() if c.isalpha()]
        if letters:
            img_cols = st.columns(min(len(letters), 6))
            for i, letter in enumerate(letters):
                found = None
                for ext in (".png", ".jpg", ".jpeg"):
                    p = os.path.join(REF_IMG_DIR, f"{letter}{ext}")
                    if os.path.exists(p):
                        found = p
                        break
                with img_cols[i % len(img_cols)]:
                    if found:
                        st.image(found, caption=letter, use_container_width=True)
                    else:
                        st.markdown(
                            f"<div class='panel' style='text-align:center;'>{letter}</div>",
                            unsafe_allow_html=True,
                        )
    else:
        st.markdown("<p class='status-muted'>No reply yet — type one above.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("##### Conversation")

if st.session_state.chat_log:
    bubbles = ""
    for entry in st.session_state.chat_log:
        cls = "bubble-you" if entry["who"] == "you" else "bubble-friend"
        label = "You (signed)" if entry["who"] == "you" else "Friend (typed)"
        bubbles += (
            f"<div class='bubble {cls}'>{entry['text']}"
            f"<div class='bubble-meta'>{label} · {entry['time']}</div></div>"
        )
    st.markdown(f"<div class='chat-scroll'>{bubbles}</div>", unsafe_allow_html=True)
else:
    st.markdown("<p class='status-muted'>Your conversation will show up here.</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# POLLING RENDERING ENGINE LOOP
# --------------------------------------------------------------------------
if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                sign_text = ctx.video_processor.sign_text
                live_word = ctx.video_processor.current_word
                mood = ctx.video_processor.mood
                mood_scores = dict(ctx.video_processor.mood_scores)

            sign_placeholder.markdown(
                f"<div class='mono-status status-teal'>{sign_text}</div>", unsafe_allow_html=True
            )
            word_placeholder.markdown(
                f"<span class='word-pill'>{live_word or '...'}</span>", unsafe_allow_html=True
            )
            mood_placeholder.markdown(
                f"<div class='mono-status status-marigold'>Mood: {mood}</div>", unsafe_allow_html=True
            )
            if mood_scores:
                rows = "".join(
                    f"<div class='mono-status status-muted' style='font-size:14px;'>"
                    f"{label}: {score:.2f}</div>"
                    for label, score in mood_scores.items()
                )
                mood_debug_placeholder.markdown(rows, unsafe_allow_html=True)
        time.sleep(0.15)
else:
    sign_placeholder.markdown(
        "<div class='mono-status status-muted'>Click Start above to begin signing.</div>",
        unsafe_allow_html=True,
    )
    word_placeholder.markdown("<span class='word-pill'>...</span>", unsafe_allow_html=True)
