import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.bboxs import grid_to_bboxs, parse_bboxs
from utils.losses import calculate_losses
from utils.metrics import calculate_accuracy


def train_model(
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        epochs: int,
        batch_size: int,
        device: str = 'cpu',
        grid_size: int = 7,
        image_size: int = 112,
        plot_every: int = 10,
        patience: int = 10,
) -> tuple[list[float]]:

    model.to(device)

    total_losses, xy_losses, wh_losses = [], [], []
    conf_obj_losses, conf_noobj_losses, class_losses, accuracies = [], [], [], []

    val_total_losses, val_xy_losses, val_wh_losses = [], [], []
    val_conf_obj_losses, val_conf_noobj_losses, val_class_losses, val_accuracies = [], [], [], []

    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in tqdm(range(1, epochs+1)):
        model.train()

        epoch_loss = []
        epoch_xy_loss = []
        epoch_wh_loss = []
        epoch_conf_obj_loss = []
        epoch_conf_noobj_loss = []
        epoch_class_loss = []
        epoch_accuracy = []

        for imgs, bboxs, labels in train_dataloader:
            imgs = imgs.to(device)
            bboxs = bboxs.to(device)
            labels = labels.to(device)

            output = model(imgs)

            pred_bboxs, pred_labels = grid_to_bboxs(output, 0.5)
            accuracy, _, _, _ = calculate_accuracy(
                pred_bboxs, bboxs, pred_labels, labels)

            bboxs = parse_bboxs(
                bboxs.reshape(-1, 4), grid_size, image_size, device)
            labels = F.one_hot(labels)
            
            losses = calculate_losses(
                output, bboxs, labels, device)
            loss = torch.sum(torch.stack(losses))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss) / batch_size)
            epoch_xy_loss.append(float(losses[0]) / batch_size)
            epoch_wh_loss.append(float(losses[1]) / batch_size)
            epoch_conf_obj_loss.append(float(losses[2]) / batch_size)
            epoch_conf_noobj_loss.append(float(losses[3]) / batch_size)
            epoch_class_loss.append(float(losses[4]) / batch_size)
            epoch_accuracy.append(accuracy)

        total_losses.append(np.mean(epoch_loss))
        xy_losses.append(np.mean(epoch_xy_loss))
        wh_losses.append(np.mean(epoch_wh_loss))
        conf_obj_losses.append(np.mean(epoch_conf_obj_loss))
        conf_noobj_losses.append(np.mean(epoch_conf_noobj_loss))
        class_losses.append(np.mean(epoch_class_loss))
        accuracies.append(np.mean(epoch_accuracy))

        model.eval()
        with torch.no_grad():
            val_loss, val_xy_loss, val_wh_loss = [], [], []
            val_conf_obj_loss, val_conf_noobj_loss, val_class_loss, val_accuracy = [], [], [], []

            for imgs, bboxs, labels in valid_dataloader:
                imgs, bboxs, labels = imgs.to(device), bboxs.to(device), labels.to(device)
                output = model(imgs)

                pred_bboxs, pred_labels = grid_to_bboxs(output, 0.5)
                accuracy, _, _, _ = calculate_accuracy(pred_bboxs, bboxs, pred_labels, labels)

                bboxs = parse_bboxs(bboxs.reshape(-1, 4), grid_size, image_size, device)
                labels = F.one_hot(labels)

                losses = calculate_losses(output, bboxs, labels, device)
                loss = torch.sum(torch.stack(losses))

                val_loss.append(float(loss) / batch_size)
                val_xy_loss.append(float(losses[0]) / batch_size)
                val_wh_loss.append(float(losses[1]) / batch_size)
                val_conf_obj_loss.append(float(losses[2]) / batch_size)
                val_conf_noobj_loss.append(float(losses[3]) / batch_size)
                val_class_loss.append(float(losses[4]) / batch_size)
                val_accuracy.append(accuracy)

        val_total_losses.append(np.mean(val_loss))
        val_xy_losses.append(np.mean(val_xy_loss))
        val_wh_losses.append(np.mean(val_wh_loss))
        val_conf_obj_losses.append(np.mean(val_conf_obj_loss))
        val_conf_noobj_losses.append(np.mean(val_conf_noobj_loss))
        val_class_losses.append(np.mean(val_class_loss))
        val_accuracies.append(np.mean(val_accuracy))

        if epoch % plot_every == 0:
            print(f'EPOCH: {epoch}, '
                  f'TRAIN ACC@50: {accuracies[-1]:.4f}, TRAIN LOSS: {total_losses[-1]:.4f}, '
                  f'VAL ACC@50: {val_accuracies[-1]:.4f}, VAL LOSS: {val_total_losses[-1]:.4f}')

        if np.mean(np.mean(val_loss)) < best_loss:
            best_loss = np.mean(val_loss)
            early_stopping_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'./models/{model.__class__.__name__}.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print('fEarly stopping the model due to no improvement for {patience} epochs')
                break

        lr_scheduler.step()

    return (
        total_losses, xy_losses, wh_losses, conf_obj_losses, conf_noobj_losses, class_losses, accuracies,
        val_total_losses, val_xy_losses, val_wh_losses, val_conf_obj_losses, val_conf_noobj_losses, val_class_losses, val_accuracies
    )
