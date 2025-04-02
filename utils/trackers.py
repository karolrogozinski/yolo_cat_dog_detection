import cv2
import numpy as np
import torch

from utils.bboxs import grid_to_bboxs


def crop_center_square(frame: np.ndarray) -> np.ndarray:
    """
    Crop the center square part of the frame.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    x_start = (w - side) // 2
    y_start = (h - side) // 2
    return frame[y_start:y_start + side, x_start:x_start + side]


def preprocess_frame(frame: np.ndarray, input_size: int = 112) -> torch.Tensor:
    """
    Preprocess the frame by resizing it and normalizing it into a tensor.
    """
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return frame_tensor.unsqueeze(0)


def draw_bboxes(frame: np.ndarray, bboxs: list[torch.Tensor],
                labels: list[torch.Tensor], input_size: int = 112) -> None:
    """
    Draw bounding boxes and labels on the frame.
    """
    h_orig, w_orig = frame.shape[:2]

    for bbox, label in zip(bboxs, labels):
        for i in range(bbox.shape[0]):
            xmin, ymin, xmax, ymax, score = [coord for coord in bbox[i]]
            lbl = label[i]

            xmin = int(xmin * w_orig / input_size)
            ymin = int(ymin * h_orig / input_size)
            xmax = int(xmax * w_orig / input_size)
            ymax = int(ymax * h_orig / input_size)

            if lbl == 1:
                color = (203, 192, 255)  # pink (BGR)
                label_text = f'Dog: {round(float(score), 4)}'
            else:
                color = (255, 200, 100)  # light blue (BGR)
                label_text = f'Cat: {round(float(score), 4)}'

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (xmin, ymin - text_height - 5),
                          (xmin + text_width, ymin), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def predict_bboxes(model: torch.nn.Module, frame: torch.Tensor,
                   obj_th: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Predict bounding boxes and labels for a given frame.
    """
    model.eval()
    with torch.no_grad():
        output = model(frame)
        pred_bboxs, pred_labels = grid_to_bboxs(output, obj_th)
    return pred_bboxs, pred_labels


def process_video(model: torch.nn.Module, obj_th: float,
                  input_video_path: str = "data/videos/cats_dogs.mov",
                  output_video_path: str = "data/videos/cats_dogs_output.avi") -> None:
    """
    Process the video by predicting and drawing bounding boxes on each frame.
    """
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_video_path}")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    side = min(frame_width, frame_height)

    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 15, (side, side))

    i = 0
    cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if i % 3 == 0:
            cropped_frame = crop_center_square(frame)
            input_tensor = preprocess_frame(cropped_frame)

            bboxs, labels = predict_bboxes(model, input_tensor, obj_th)
            draw_bboxes(cropped_frame, bboxs, labels)

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
            output.write(cropped_frame)
            cnt += 1

        i += 1

    cap.release()
    output.release()
    cv2.destroyAllWindows()
