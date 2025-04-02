import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CatDogYOLOv1(nn.Module):
    """
    YOLOv1-like CNN model for object detection.
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(1568, 512)
        self.fc2 = nn.Linear(512, 343)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        x = x.view(x.shape[0], 7, 7, -1)
        return x


class CatDogYOLOv1Improved(nn.Module):
    """
    Improved version of YOLOv1 with additional layers.
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 343)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.1)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))

        x = x.view(x.shape[0], 7, 7, -1)
        return x


class CatDogYOLOv1ResBB(nn.Module):
    """
    YOLOv1 model with a ResNet18 backbone.
    """
    def __init__(self, hot_layers: int = 5) -> None:
        super().__init__()

        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in list(self.backbone.parameters())[-hot_layers:]:
            param.requires_grad = True

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 343)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))

        x = x.view(x.shape[0], 7, 7, -1)
        return x
