import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from utils.bboxs import grid_to_bboxs
from utils.metrics import calculate_dataloader_mAP, compute_confusion_matrix


def visualize_batch(dataloader, number_of_images: int) -> None:
    """
    Function to visualize a batch of images
    along with their ground truth bounding boxes and labels.
    """
    images, bboxes, labels = next(iter(dataloader))

    images = images[:number_of_images]
    bboxes = bboxes[:number_of_images]
    labels = labels[:number_of_images]

    _, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]

    for i, (img, bbox, label) in enumerate(zip(images, bboxes, labels)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bbox, label):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(xmin, ymin - 5,
                         f'Label: {lbl.item()}', color='red', fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.5))
        axes[i].axis('off')

    plt.show()


def plot_predictions(model, dataloader, confidence_th: float, n_images: int = 6) -> None:
    """
    Plot predictions of the model for a batch of images.
    """
    model.eval()
    for imgs, bboxs, labels in dataloader:
        output = model(imgs)
        break

    bboxs, labels = grid_to_bboxs(output, confidence_th)

    imgs = imgs[:n_images]
    bboxs = bboxs[:n_images]
    labels = labels[:n_images]

    _, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    if len(imgs) == 1:
        axes = [axes]

    for i, (img, bbox, label) in enumerate(zip(imgs, bboxs, labels)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bbox, label):
            xmin, ymin, xmax, ymax, score = box.tolist()
            if lbl.item() == 1:
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=3, edgecolor='pink', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(xmin, ymin - 5, f'Dog: {round(score, 4)}', color='pink', fontsize=12,
                             bbox=dict(facecolor='black'))
            else:
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=3, edgecolor='lightblue', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(xmin, ymin - 5, f'Cat: {round(score, 4)}', color='lightblue', fontsize=12,
                             bbox=dict(facecolor='black'))

        axes[i].axis('off')

    plt.show()


def plot_mAPs(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader
) -> float:
    """
    Plot the mAP for train, validation, and test datasets.
    """
    train_mAPs = calculate_dataloader_mAP(model, train_dataloader)
    valid_mAPs = calculate_dataloader_mAP(model, valid_dataloader)
    test_mAPs = calculate_dataloader_mAP(model, test_dataloader)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, mAPs, title, color in zip(
        axes, [train_mAPs, valid_mAPs, test_mAPs], 
        ['Train mAP@50', 'Valid mAP@50', 'Test mAP@50'], 
        ['pink', 'lightblue', 'red']
    ):
        ax.plot(mAPs.keys(), mAPs.values(), color=color, linewidth=3)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('Objectness threshold')
        ax.set_ylabel('mAP')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return max(test_mAPs, key=test_mAPs.get)


def plot_train_metrics(
    total_losses: list[float],
    xy_losses: list[float],
    wh_losses: list[float],
    conf_obj_losses: list[float],
    conf_noobj_losses: list[float],
    class_losses: list[float],
    accuracies: list[float],
    val_total_losses: list[float],
    val_xy_losses: list[float],
    val_wh_losses: list[float],
    val_conf_obj_losses: list[float],
    val_conf_noobj_losses: list[float],
    val_class_losses: list[float],
    val_accuracies: list[float]
) -> None:
    """
    Plot training and validation metrics including losses and accuracies.
    """
    epochs = range(len(total_losses))

    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_configs = [
        ("Total Loss", total_losses, val_total_losses, axes[0, 0]),
        ("XY Loss", xy_losses, val_xy_losses, axes[0, 1]),
        ("WH Loss", wh_losses, val_wh_losses, axes[0, 2]),
        ("Class Loss", class_losses, val_class_losses, axes[1, 0]),
        ("Accuracy@50", accuracies, val_accuracies, axes[1, 1]),
    ]

    for title, train_metric, val_metric, ax in plot_configs:
        ax.plot(epochs, train_metric, color="pink", linewidth=3, label="Train")
        ax.plot(epochs, val_metric, color="lightblue", linewidth=3, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.grid(axis='y', alpha=.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    ax = axes[1, 2]
    ax.plot(epochs, conf_obj_losses, color="pink",
            linewidth=3, label="Train Obj")
    ax.plot(epochs, val_conf_obj_losses, color="lightblue",
            linewidth=3, label="Val Obj")
    ax.plot(epochs, conf_noobj_losses, color="pink", linestyle="dotted",
            linewidth=3, label="Train Noobj")
    ax.plot(epochs, val_conf_noobj_losses, color="lightblue", linestyle="dotted",
            linewidth=3, label="Val Noobj")

    ax.set_title("Confidence Loss (Obj & Noobj)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.grid(axis='y', alpha=.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, obj_th: float) -> None:
    """
    Plot the confusion matrix for model predictions on a given dataset.
    """
    conf_matrix = compute_confusion_matrix(model, dataloader, obj_threshold=obj_th)
    classes = ["Cat", "Dog", "None"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=sns.cubehelix_palette(as_cmap=True),
                xticklabels=classes, yticklabels=classes, linewidths=1, linecolor='white')
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.show()


def predict_plot_single_image(model, img, obj_th: float) -> None:
    """
    Plot a single image with predictions made by the model.
    """
    model.eval()
    output = model(img)

    bboxs, labels = grid_to_bboxs(output, obj_th)

    _, ax = plt.subplots(figsize=(15, 5))

    img = img[0].permute(1, 2, 0).numpy()
    ax.imshow(img)

    for box, lbl in zip(bboxs[0], labels[0]):
        try:
            xmin, ymin, xmax, ymax, score = box.tolist()
        except ValueError:
            continue

        color = 'pink' if lbl.item() == 1 else 'lightblue'
        label_text = 'Dog' if lbl.item() == 1 else 'Cat'

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f'{label_text}: {round(score, 4)}',
                color=color, fontsize=12, bbox=dict(facecolor='black'))

    ax.axis('off')
    plt.show()
