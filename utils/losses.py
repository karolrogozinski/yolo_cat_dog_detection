import torch
from torch import Tensor


def calculate_losses(
    predictions: Tensor, 
    bboxs: Tensor,
    labels: Tensor,
    device: str = 'cpu',
    coord_param: float = 5,
    noobj_param: float = 0.5
) -> Tensor:
    """
    Calculate the losses for YOLOv1 model, including losses for bounding box coordinates, 
    width/height, confidence, and classification.
    """
    batch_size = predictions.shape[0]
    grid_size = predictions.shape[1]

    mask_bboxs = bboxs.unsqueeze(1)
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(
        mask_bboxs[:, :, 0]).to(device)

    obj_mask = torch.zeros((batch_size, grid_size, grid_size),
                           dtype=torch.float32).to(device)
    obj_mask[batch_idx, mask_bboxs[:, :, 0].long(), mask_bboxs[:, :, 1].long()] = 1

    noobj_mask = torch.ones((batch_size, grid_size, grid_size),
                            dtype=torch.float32).to(device)
    noobj_mask[batch_idx, mask_bboxs[:, :, 0].long(), mask_bboxs[:, :, 1].long()] = 0

    losses = [
        coord_param * xy_loss(predictions, bboxs, obj_mask),
        coord_param * wh_loss(predictions, bboxs, obj_mask),
        conf_loss(predictions, obj_mask, torch.tensor(1).to(device)),
        noobj_param * conf_loss(predictions, noobj_mask, torch.tensor(0).to(device)),
        class_loss(predictions, labels, obj_mask)
    ]

    return losses


def xy_loss(predictions: Tensor, bboxs: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate the loss for the x and y coordinates of the bounding box.
    """
    bboxs_x = bboxs[:, 2][:, None, None]
    bboxs_x.expand(-1, 7, 7)

    bboxs_y = bboxs[:, 3][:, None, None]
    bboxs_y.expand(-1, 7, 7)

    x_loss = torch.square(predictions[:, :, :, 0] - bboxs_x) * mask
    y_loss = torch.square(predictions[:, :, :, 1] - bboxs_y) * mask

    return torch.sum(x_loss) + torch.sum(y_loss)


def wh_loss(predictions: Tensor, bboxs: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate the loss for the width and height of the bounding box.
    """
    bboxs_w = bboxs[:, 4][:, None, None]
    bboxs_w.expand(-1, 7, 7)

    bboxs_h = bboxs[:, 5][:, None, None]
    bboxs_h.expand(-1, 7, 7)

    w_loss = torch.square(torch.sqrt(predictions[:, :, :, 2]
                                     ) - torch.sqrt(bboxs_w)) * mask
    h_loss = torch.square(torch.sqrt(predictions[:, :, :, 3]
                                     ) - torch.sqrt(bboxs_h)) * mask

    return torch.sum(w_loss) + torch.sum(h_loss)


def conf_loss(predictions: Tensor, mask: Tensor, is_conf: int) -> Tensor:
    """
    Calculate the confidence loss.
    """
    conf_loss_value = torch.square(predictions[:, :, :, 4] - is_conf) * mask
    return torch.sum(conf_loss_value)


def class_loss(predictions: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate the class loss.
    """
    class_0 = labels[:, :, 0].flatten()[:, None, None]
    class_0.expand(-1, 7, 7)
    class_0_loss = torch.square(predictions[:, :, :, 5] - class_0) * mask

    try:
        class_1 = labels[:, :, 1].flatten()[:, None, None]
        class_1.expand(-1, 7, 7)
        class_1_loss = torch.square(predictions[:, :, :, 6] - class_1) * mask
    except IndexError:
        class_1_loss = torch.square(predictions[:, :, :, 6]) * mask

    return torch.sum(class_0_loss) + torch.sum(class_1_loss)
