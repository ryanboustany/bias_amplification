import torch
import torch.nn as nn
import sys

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True, input_channels=3, input_height=64, input_width=64):
        super(LeNet5, self).__init__()
        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        self.features = self._make_layers()

        self.flatten_size = self._calculate_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 120),
            nn.ReLU(True),
            #nn.BatchNorm1d(120) if self.batch_norm else nn.Identity(),
            nn.Linear(120, 84),
            nn.ReLU(True),
            #nn.BatchNorm1d(84) if self.batch_norm else nn.Identity(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

    def _make_layers(self):
        layers = [
            nn.Conv2d(in_channels=self.input_channels, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(6) if self.batch_norm else nn.Identity(),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(16) if self.batch_norm else nn.Identity(),
        ]
        return nn.Sequential(*layers)

    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            output = self.features(dummy_input)
            flatten_size = output.numel()  
            return flatten_size