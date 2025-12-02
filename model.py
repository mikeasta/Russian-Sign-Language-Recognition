import torch
from torch import nn

class RSLRmodel(nn.Module):
    def __init__(self):
        super().__init__()

        # 3x64x64 => 32x32x32
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # 32x32x32 => 32x16x16
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # 32x16x16 => 32x8x8
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # 64x8x8 => 2048 => 25
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=.5),
            nn.Linear(
                in_features=2048,
                out_features=25,
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        return self.classifier(x3)