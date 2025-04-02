import torch
from mean_average_precision import MetricBuilder
import numpy as np
from utils.bboxs import grid_to_bboxs


def calculate_mAP(
    pred_bboxs: list[list[torch.Tensor]], 
    bboxs: list[list[torch.Tensor]], 
    pred_labels: list[list[int]], 
    labels: list[list[int]], 
    num_classes: int = 2, 
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean Average Precision (mAP) for the given predicted
    and ground truth bounding boxes and labels.
    """
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d",async_mode=False, num_classes=num_classes)

    for i in range(len(pred_bboxs)):
        preds = np.array([
            pred_bboxs[i][j].detach().tolist() + [pred_labels[i][j]]
            for j in range(len(pred_bboxs[i]))
        ])
        try:
            preds[:, [4, 5]] = preds[:, [5, 4]]
        except IndexError:
            continue

        gts = np.array([
            bboxs[i][j].detach().tolist() + [labels[i][j], 0, 0]
            for j in range(len(bboxs[i]))
        ])

        metric_fn.add(preds, gts)

    mAP = metric_fn.value(iou_thresholds=iou_threshold)['mAP']
    return mAP


def calculate_dataloader_mAP(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader
) -> dict[float, float]:
    """
    Calculate mAP for a given model and dataloader.
    """
    mAP_scores = {i / 10: [] for i in range(11)}

    model.eval()
    for imgs, bboxs, labels in dataloader:
        output = model(imgs)

        for conf_th in mAP_scores.keys():
            pred_bboxs, pred_labels = grid_to_bboxs(output, conf_th)
            mAP = calculate_mAP(pred_bboxs, bboxs, pred_labels, labels)
            mAP_scores[conf_th].append(mAP)

    for key, value in mAP_scores.items():
        mAP_scores[key] = np.mean(value)

    return mAP_scores


def iou(box1: list[float], box2: list[list[float]]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2[0]

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0


def calculate_accuracy(
    pred_bboxs: list[list[torch.Tensor]],
    bboxs: list[list[torch.Tensor]],
    pred_labels: list[list[int]],
    labels: list[list[int]],
    iou_threshold: float = 0.5
) -> tuple[float, int, int, int]:
    """
    Calculate accuracy based on bounding boxes.
    """
    TP, FP, FN = 0, 0, 0

    for i in range(len(pred_bboxs)):
        matched = 0

        for j, pred_box in enumerate(pred_bboxs[i]):

            pred_label = pred_labels[i][j]
            found_match = False

            gt_label = labels[i]
            if iou(pred_box[:4], bboxs[i]) >= iou_threshold and \
               pred_label == gt_label and not matched:
                TP += 1
                matched = 1
                found_match = True

            if not found_match:
                FP += 1

        FN += 1 if not matched else 0

    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    return accuracy, TP, FP, FN


def compute_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    iou_threshold: float = 0.5,
    obj_threshold: float = 0.5
) -> np.ndarray:
    """
    Compute the confusion matrix for the model on a given dataloader.
    """
    confusion_matrix = np.zeros((3, 3), dtype=int)  # 3 classes: Cat (0), Dog (1), None (2)

    model.eval()
    with torch.no_grad():
        for imgs, bboxs, labels in dataloader:
            outputs = model(imgs)
            pred_bboxs, pred_labels = grid_to_bboxs(outputs, obj_threshold)

            for i in range(len(labels)):
                gt_label = labels[i].item()
                matched = False

                if len(pred_bboxs[i]) > 0:

                    for j, pred_box in enumerate(pred_bboxs[i]):
                        pred_label = pred_labels[i][j]

                        iou_score = iou(pred_box[:4], bboxs[i]) if len(bboxs[i]) > 0 else 0
                        
                        if iou_score >= iou_threshold and pred_label == gt_label:
                            confusion_matrix[pred_label][gt_label] += 1
                            matched = True

                        elif iou_score >= iou_threshold and pred_label != gt_label:
                            confusion_matrix[pred_label][gt_label] += 1

                        else:
                            confusion_matrix[pred_label][2] += 1

                if not matched:
                    confusion_matrix[2][gt_label] += 1

    return confusion_matrix
