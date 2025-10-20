import cv2

# --- VISUALIZATION SETTINGS ---
BOX_COLOR = (0, 255, 0)  # Green color for the bounding box (in BGR format)
TEXT_COLOR = (255, 255, 255)  # White color for the text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2


def draw_detections(frame, results):
    """
    Draws bounding boxes and labels on a frame based on YOLO detection results.

    Args:
        frame (np.ndarray): The image/frame to draw on.
        results: The YOLO results object containing the detections.

    Returns:
        np.ndarray: The frame with detections drawn on it.
    """
    # Get the bounding boxes, confidence scores, and class IDs
    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confs = results.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
    class_names = results.names  # Dictionary of class ID -> class name

    # Loop through each detection
    for box, conf, cls_id in zip(boxes, confs, class_ids):
        # Unpack bounding box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Get the class name
        label = class_names[int(cls_id)]

        # Create the label text with class name and confidence score
        label_text = f"{label}: {conf:.2f}"

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, FONT_THICKNESS)

        # --- Draw the text with a filled background for better visibility ---
        # Calculate text size to create a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)

        # Position the background rectangle
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - baseline
        text_bg_x2 = x1 + text_width
        text_bg_y2 = y1

        # Draw the filled rectangle for the text background
        cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), BOX_COLOR, -1)

        # Draw the text on top of the background
        cv2.putText(frame, label_text, (x1, y1 - baseline), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    return frame