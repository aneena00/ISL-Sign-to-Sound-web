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
        img =
