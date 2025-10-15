import torch
import torch.nn as nn
import sys 

# Depthwise separable conv block
def conv_dw(in_planes, out_planes, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

class MobileNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_height=32, input_width=32):
        super(MobileNet, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(512, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
