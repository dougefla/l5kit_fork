from black import out
import torch.nn as nn
from torchvision.models.resnet import resnet50
from models.LMM import LMM


class PlanningNet():
    def __init__(
        self,
        model_arch: str,
        input_channels: int,
        predict_frames: int,
        pretrained: bool,
    ):
        self.model_arch = model_arch
        self.input_channels = input_channels
        self.predict_frames = predict_frames
        self.pretrained = pretrained
    def get_model(self):
        if self.model_arch == 'resnet_50':
            model = resnet50(pretrained=self.pretrained)
            model.fc = nn.Linear(in_features=2048, out_features=3*self.predict_frames)
            return model
        elif self.model_arch == 'efficient_net_b3':
            return LMM(input_channels=3, output_channels=3*self.predict_frames)