import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False, norm_layer=None):
        super(_DenseLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        self.add_module('norm1', norm_layer(num_input_features))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', norm_layer(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features))) 
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False, norm_layer=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm_layer=norm_layer
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=None, relu_fn=nn.ReLU):
        super(_Transition, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', relu_fn(True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10,
                 memory_efficient=False, norm_layer=None,
                 input_channels=3, input_height=32, input_width=32):
        super(DenseNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', norm_layer(num_init_features)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm_layer=norm_layer
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    norm_layer=norm_layer, relu_fn=nn.ReLU)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', norm_layer(num_features))
        self.relu = nn.ReLU(True)
        self.flatten_size = self._calculate_flatten_size()
        self.classifier = nn.Linear(self.flatten_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            output = self.features(dummy_input)
            flatten_size = output.numel()
            return flatten_size

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model

def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)

def densenet201(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)