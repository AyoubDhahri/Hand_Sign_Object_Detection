import time
import cv2
import numpy as np
from ultralytics import YOLO

# --- 🚀 CONFIGURATION 🚀 ---
# Set the path to the model you want to test.
MODEL_PATH = 'models/best.pt'

# Set the path to a realistic test image.
# IMPORTANT: This image should be similar in size to your webcam feed (e.g., 640x480 or 1280x720).
# Create a test image named 'test_hand_sign.jpg' and place it in your 'assets' folder.
IMAGE_PATH = 'assets/test.jpg'


# -------------------------

def measure_prediction_time():
    """
    Loads the YOLO model, performs a warm-up, and then times a
    single prediction on a test image to measure performance.
    """
    try:
        # 1. Load the YOLO model from the specified path
        print(f"🔄 Loading YOLO model from '{MODEL_PATH}'...")
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully.")
        print("-" * 40)

        # 2. Load the test image using OpenCV
        print(f"🔄 Loading test image from '{IMAGE_PATH}'...")
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print(f"❌ ERROR: Could not load the image at '{IMAGE_PATH}'.")
            print("Please ensure the file exists and is a valid image.")
            return
        print(f"✅ Test image loaded successfully (Dimensions: {image.shape[1]}x{image.shape[0]}).")
        print("-" * 40)

        # 3. Perform a "warm-up" prediction.
        # The very first inference is often much slower because the model's components
        # are being loaded into memory (e.g., onto the GPU). Running a single, untimed
        # prediction first gives a much more accurate measurement for subsequent runs.
        print("🔥 Performing a warm-up prediction...")
        # We can use a simple black image for the warm-up.
        dummy_image = np.zeros_like(image)
        model(dummy_image, verbose=False)
        print("✅ Warm-up complete.")
        print("-" * 40)

        # 4. Time the actual prediction on your real test image
        print("⏱️  Running timed prediction...")

        start_time = time.time()
        # The main event: run the model on the image. verbose=False stops it from printing detection details.
        results = model(image, verbose=False)
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time
        num_detections = len(results[0].boxes)

        # 5. Print the final, user-friendly report
        print("\n--- 🚀 Detection Performance Report 🚀 ---")
        print(f"🔹 Time for a single prediction: {duration * 1000:.2f} ms")
        print(f"🔹 Theoretical Max FPS:           {1 / duration:.2f} FPS")
        print(f"🔹 Detections found in image:     {num_detections}")
        print("------------------------------------------\n")
        print("💡 Use this 'ms' value to help decide the 'detection_interval' in your Streamlit app.")
        print("   A good interval is one that gives your app time to breathe between predictions.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check that your model and image paths are correct and all libraries are installed.")


if __name__ == "__main__":
    measure_prediction_time()