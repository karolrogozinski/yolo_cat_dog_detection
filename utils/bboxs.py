import torch


def parse_bboxs(bboxs: torch.tensor, grid_size: int,
                img_size: int, device: str = 'cpu') -> torch.Tensor:
    """
    Parse raw bounding boxes from xmin, ymin, xmax, ymax into YOLO format.
    """
    cell_size = img_size / grid_size
    yolo_bboxs = []

    for xmin, ymin, xmax, ymax in bboxs:
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        cell_x = min(grid_size - 1, int(x // cell_size))
        cell_y = min(grid_size - 1, int(y // cell_size))

        x_norm = (x / cell_size) - cell_x
        y_norm = (y / cell_size) - cell_y
        w_norm = w / img_size
        h_norm = h / img_size

        yolo_bboxs.append((cell_x, cell_y, x_norm, y_norm, w_norm, h_norm))

    return torch.tensor(yolo_bboxs, dtype=torch.float32).to(device)


def grid_to_bboxs(predictions: torch.Tensor, confidence_threshold: float,
                  img_size: int = 112, grid_size: int = 7):
    """
    Convert YOLO grid predictions into bounding boxes.
    """
    batch_size = predictions.shape[0]
    cell_size = img_size / grid_size
    
    all_bboxs = []
    all_labels = []
    
    for b in range(batch_size):  
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).T
        grid_y = torch.arange(grid_size).repeat(grid_size, 1)

        x = (grid_x + predictions[b, :, :, 0]) * cell_size
        y = (grid_y + predictions[b, :, :, 1]) * cell_size
        w = predictions[b, :, :, 2] * img_size
        h = predictions[b, :, :, 3] * img_size

        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2

        class_probs = predictions[b, :, :, 5:]
        labels = class_probs.argmax(dim=-1)

        mask = predictions[b, :, :, 4] > confidence_threshold

        masked_xmin = xmin[mask]
        masked_ymin = ymin[mask]
        masked_xmax = xmax[mask]
        masked_ymax = ymax[mask]
        masked_score = predictions[b, :, :, 4][mask]
        masked_labels = labels[mask]

        bboxs = torch.stack([masked_xmin, masked_ymin, masked_xmax,
                             masked_ymax, masked_score], dim=1)
        all_bboxs.append(bboxs)
        all_labels.append(masked_labels)

    return all_bboxs, all_labels
