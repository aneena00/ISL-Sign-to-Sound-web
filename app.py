"""
Sign2Sound — Streamlit Web App
================================
Two-way ISL <-> Speech translator. Works locally AND once deployed publicly,
because the camera is captured in the VISITOR's browser (via streamlit-webrtc)
and speech is spoken by the VISITOR's browser (via the Web Speech API) —
nothing depends on hardware attached to the server.

Run with:
    streamlit run streamlit_app.py

Requires these files in the SAME folder:
    model.py
    mediapipe_extractor.py
    calculate_emotion.py
    hand_landmarker.task
    isl_model.pth          (your trained weights)
    reference_images/      (A.png, B.png, ... one image per letter)
"""

import os
import io
import json
import time
import threading
from datetime import datetime

import av
import cv2
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_local_storage import LocalStorage

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

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
REF_IMG_DIR = ""  # letter images live alongside app.py etc. at the repo root
LABELS_FILE = "labels.json"
DEFAULT_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # no "R"

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

@st.cache_resource(ttl=3000)  # refresh before Twilio's ~1hr token expiry
def get_rtc_configuration():
    """
    Twilio's Network Traversal Service gives a private, reliable TURN
    endpoint (vs. the free public Open Relay demo servers, which turned
    out to be too unreliable to actually establish a connection). Falls
    back to the old public servers if Twilio secrets aren't set yet, so
    the app doesn't hard-crash — but expect that fallback to be flaky.
    """
    try:
        from twilio.rest import Client
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        token = client.tokens.create()
        return RTCConfiguration({"iceServers": token.ice_servers, "iceTransportPolicy": "relay"})
    except Exception as e:
        print(f"[Warning] Twilio ICE servers unavailable, falling back to public TURN: {e}")
        return RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": ["turns:openrelay.metered.ca:443?transport=tcp"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                ],
                "iceTransportPolicy": "relay",
            }
        )


# (RTC_CONFIGURATION is assigned just below, after st.set_page_config())

# --------------------------------------------------------------------------
# PAGE + THEME
# --------------------------------------------------------------------------
st.set_page_config(page_title="Sign2Sound", layout="wide", page_icon="🤟")
RTC_CONFIGURATION = get_rtc_configuration()

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
# SESSION STATE (main-thread, per-browser-tab state)
# --------------------------------------------------------------------------
defaults = {
    "received_text": "",
    "chat_log": [],   # list of {"who": "you"/"friend", "text": ..., "time": ...}
    "last_audio_bytes": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------------------------------
# PERSISTENT CHAT HISTORY (saved in the VISITOR's own browser localStorage —
# private to their device, no server-side database needed). Custom
# components like this return their value asynchronously, so the saved
# history may only become available a render or two after the page first
# loads — that's expected, not a bug.
# --------------------------------------------------------------------------
local_storage = LocalStorage()
CHAT_STORAGE_KEY = "sign2sound_chat_log"

if not st.session_state.get("_chat_loaded_from_storage"):
    try:
        saved_chat = local_storage.getItem(CHAT_STORAGE_KEY)
        if saved_chat:
            st.session_state.chat_log = json.loads(saved_chat)
    except Exception:
        pass
    st.session_state._chat_loaded_from_storage = True


def save_chat_to_browser():
    try:
        local_storage.setItem(CHAT_STORAGE_KEY, json.dumps(st.session_state.chat_log))
    except Exception:
        pass

# --------------------------------------------------------------------------
# CACHED, READ-ONLY RESOURCES (safe to share across sessions)
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
# BROWSER-SIDE SPEECH (plays on whoever is viewing the page, local or remote)
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
# Each browser tab/visitor gets its own instance (own hand extractor, own
# emotion detector, own buffers) — important so concurrent users don't
# interfere with each other's predictions.
# --------------------------------------------------------------------------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        # Keep this constructor as fast as possible — it runs WHILE the
        # WebRTC connection is still being negotiated. Loading two
        # MediaPipe models here was likely taking longer than the
        # connection handshake/ICE candidates allow for on a slow shared
        # cloud CPU, so the connection died right around when loading
        # finished. Everything heavy is now deferred to _ensure_loaded(),
        # called from recv() — by then the connection already succeeded,
        # so a slow first frame or two is harmless instead of fatal.
        self.lock = threading.Lock()
        self.extractor = None
        self.emotion_detector = None
        self._models_loaded = False

        self.sequence = []
        self.stability_buffer = []
        self.current_word = ""
        self.last_added_letter = ""
        self.last_addition_time = 0.0

        self.sign_text = "Loading models..."
        self.mood = "Neutral"
        self.mood_scores = {}
        self.frame_count = 0

    def _ensure_loaded(self):
        if self._models_loaded:
            return
        self.extractor = MediaPipeFeatureExtractor()
        # Mood detection is a nice-to-have — if its model can't be loaded
        # (e.g. no internet to fetch face_landmarker.task), disable it
        # quietly rather than taking the whole video stream down with it.
        if EmotionDetector is not None:
            try:
                self.emotion_detector = EmotionDetector()
            except Exception as e:
                print(f"[Warning] Mood detection disabled — could not load face model: {e}")
        self._models_loaded = True

    def recv(self, frame):
        self._ensure_loaded()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        self.frame_count += 1

        # Mood detection runs its own full face-mesh model — doing that on
        # every single frame, on top of hand tracking + the LSTM, is more
        # than a free cloud CPU can keep up with in real time, which is
        # exactly what causes the connection to stall and get torn down a
        # couple seconds in. Mood doesn't need to refresh that often, so
        # only recompute it every 8th frame and reuse the last value
        # otherwise.
        mood = self.mood
        mood_scores = self.mood_scores
        if self.emotion_detector is not None and self.frame_count % 8 == 0:
            try:
                mood, mood_scores = self.emotion_detector.calculate_with_scores(img)
            except Exception:
                mood = "Neutral"
        elif self.emotion_detector is None:
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
# HEADER
# --------------------------------------------------------------------------
st.markdown("<div class='hero-title'>Sign2Sound</div>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>A live bridge between ISL signs and spoken word.</p>", unsafe_allow_html=True)
st.markdown("<div class='bridge-line'></div>", unsafe_allow_html=True)

col_cam, col_reply = st.columns(2)

# --------------------------------------------------------------------------
# LEFT COLUMN — camera + your signs
# --------------------------------------------------------------------------
with col_cam:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("##### Your Camera")

    ctx = webrtc_streamer(
        key="sign2sound",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 480}}, "audio": False},
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
                save_chat_to_browser()
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

# --------------------------------------------------------------------------
# RIGHT COLUMN — reply + now translating
# --------------------------------------------------------------------------
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
        save_chat_to_browser()

    st.markdown(
        "<p class='status-muted' style='margin:10px 0 4px;'>— or speak it —</p>",
        unsafe_allow_html=True,
    )

    if audio_recorder is None or sr is None:
        st.markdown(
            "<p class='status-muted'>Voice reply needs audio-recorder-streamlit "
            "and SpeechRecognition (see requirements.txt).</p>",
            unsafe_allow_html=True,
        )
    else:
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
                except sr.UnknownValueError:
                    heard = None
                    st.warning("Didn't catch that clearly — try speaking closer to the mic.")
                except sr.RequestError as e:
                    heard = None
                    st.warning(f"Speech recognition service error: {e}")
                except Exception as e:
                    heard = None
                    st.warning(f"Could not process that recording: {e}")

            if heard:
                st.session_state.received_text = heard
                speak_in_browser(heard)
                st.session_state.chat_log.append(
                    {"who": "friend", "text": heard, "time": datetime.now().strftime("%H:%M")}
                )
                save_chat_to_browser()

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
                        st.image(found, caption=letter, width="stretch")
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
chead1, chead2 = st.columns([4, 1])
chead1.markdown("##### Conversation")
if chead2.button("Clear History", use_container_width=True):
    st.session_state.chat_log = []
    save_chat_to_browser()
    st.rerun()

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
# LIVE POLLING LOOP — pulls status from the video processor and updates
# the placeholders above. Interrupted automatically whenever a button is
# clicked elsewhere on the page (Streamlit cancels this run and starts a
# fresh one), which is what makes Start/Stop/Speak/Clear all work.
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
