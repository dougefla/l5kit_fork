from timm.models.layers.conv2d_same import Conv2dSame
import timm
from torch import nn
import torch

class LMM(nn.Module):
    def __init__(self, input_channels, output_channels, pretrained):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.backbone = timm.create_model('tf_efficientnet_b3_ns', pretrained=pretrained)
        self.backbone.conv_stem = Conv2dSame(
            self.input_channels,
            self.backbone.conv_stem.out_channels,
            kernel_size=self.backbone.conv_stem.kernel_size,
            stride=self.backbone.conv_stem.stride,
            padding=self.backbone.conv_stem.padding,
            bias=False,
        )

        self.backbone_out_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(
                in_features=self.backbone.classifier.in_features,
                out_features=self.backbone_out_features,
            ),
        )

        self.lin_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_features=self.backbone_out_features,
                out_features=self.output_channels,
            ),
        )
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        pred = self.lin_head(x)
        return pred