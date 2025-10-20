import torch
from ultralytics import YOLO

# --- MODEL CONFIGURATION ---
MODEL_PATH = 'models/best.pt'
CONFIDENCE_THRESHOLD = 0.5  # Only detect objects with > 50% confidence


class HandSignDetector:
    """
    A class to encapsulate the YOLO model loading and inference logic.
    """

    def __init__(self, model_path=MODEL_PATH):
        """
        Initializes the HandSignDetector.

        Args:
            model_path (str): The path to the trained YOLO model file.
        """
        try:
            self.model = YOLO(model_path)
            # Optional: Move model to GPU if available for faster inference
            if torch.cuda.is_available():
                self.model.to('cuda')
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def detect_signs(self, frame):
        """
        Performs hand sign detection on a single video frame.

        Args:
            frame (np.ndarray): The input image/frame in NumPy array format.

        Returns:
            A list of detections from the YOLO model.
            Returns an empty list if the model is not loaded or no signs are found.
        """
        if self.model is None:
            return []

        # Perform inference
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # results[0] contains the detections for the first (and only) image
        return results[0]


# --- Create a singleton instance ---
# This ensures the model is loaded only once when the app starts, not on every re-run.
detector = HandSignDetector()