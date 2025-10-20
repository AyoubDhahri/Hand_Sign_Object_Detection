import streamlit as st
from PIL import Image
import av
# Import the class-based component from streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# --- Core Logic Imports ---
from core.hand_sign_detector import detector
from core.image_processing import draw_detections

# --- Icon Loading ---
try:
    sharingan_icon = Image.open("assets/icon.png")
except FileNotFoundError:
    sharingan_icon = "👁️"

# --- Page Configuration ---
# This is now the configuration for the entire app.
st.set_page_config(
    page_title="Naruto Hand Sign Detection",
    page_icon=sharingan_icon,
    layout="wide"
)

# --- Custom CSS for the Immersive Layout ---
st.markdown("""
<style>
    /* Hide Streamlit's default elements for a cleaner look */
    .main .block-container { padding: 0; margin: 0; }
    header, footer { visibility: hidden; }

    /* Ensure the columns align properly */
    div[data-testid="stHorizontalBlock"] { align-items: flex-start; }

    /* Left & Right Sidebars (Fixed Position) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1),
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) {
        position: fixed; top: 0; width: 22%; height: 100vh; overflow: hidden;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) { left: 0; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) { right: 0; }

    /* Main Content Area (Scrollable) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        margin-left: 22%; margin-right: 22%; padding: 1rem 2rem; width: 56%;
    }

    /* Make Sidebar Images Fill Their Container */
    div[data-testid="stHorizontalBlock"] > div:nth-child(n) img {
        height: 100vh; width: 100%; object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)


# --- Real-Time Video Processor with Performance Optimization ---
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_interval = 30  # Predict once every 30 frames
        self.frame_count = 0
        self.last_detections = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Run detection only at the specified interval
        if self.frame_count % self.detection_interval == 0:
            detection_results = detector.detect_signs(img)
            self.last_detections = detection_results

        # Draw the last known detections on every frame
        if self.last_detections:
            annotated_image = draw_detections(img, self.last_detections)
        else:
            annotated_image = img

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


# --- Asset Loading for Sidebars ---
try:
    left_image = Image.open("assets/naruto_image_left.png")
    right_image = Image.open("assets/naruto_image_right.jpg")
except FileNotFoundError:
    left_image = Image.new('RGB', (400, 1080), color='black')
    right_image = Image.new('RGB', (400, 1080), color='black')

# --- Main Page Layout ---
col1, col2, col3 = st.columns([1, 2, 1])

# Left Fixed Sidebar
with col1:
    st.image(left_image, use_container_width=True)

# Right Fixed Sidebar
with col3:
    st.image(right_image, use_container_width=True)

# Center Scrollable Content - This is the main interface
with col2:
    st.title("Hand Sign Detection Jutsu!")
    st.write("Click START to activate your camera for live detection.")

    webrtc_streamer(
        key="hand-sign-detector",
        video_processor_factory=HandSignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("The model is running a prediction once per second for smooth performance.")