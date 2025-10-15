import torch
import torch.nn as nn
import sys

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, batch_norm=True, num_classes=10, dropout=0.0, input_channels=3, input_height=32, input_width=32):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        self.features = self._make_layers(cfg[vgg_name])
        
        self.flatten_size = self._calculate_flatten_size()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.flatten_size, num_classes))
        
        # Initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.ReLU(True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  
        return nn.Sequential(*layers)

    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            output = self.features(dummy_input)
            flatten_size = output.numel() 
            return flatten_size
