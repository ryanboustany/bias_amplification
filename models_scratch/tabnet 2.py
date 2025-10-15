import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsemax(nn.Module):
    def forward(self, x):
        z = x - x.max(dim=-1, keepdim=True)[0]
        exp_z = torch.exp(z)
        return exp_z / exp_z.sum(dim=-1, keepdim=True)

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
        self._init()

    def forward(self, x):
        return self.net(x)

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.linear = nn.Linear(attention_dim, input_dim)  # <<< CORRECT : attention_dim -> input_dim
        self.bn = nn.BatchNorm1d(input_dim)
        self.sparsemax = Sparsemax()
        self._init()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.sparsemax(x)
        return x

    def _init(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

class TabNet(nn.Module):
    def __init__(self, cat_dims, num_continuous, n_d=32, n_a=32, n_steps=5, gamma=1.5, num_classes=2):
        super().__init__()
        self.input_dim = len(cat_dims) + num_continuous
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.initial_transform = FeatureTransformer(self.input_dim, n_d + n_a)

        self.attentive_transformers = nn.ModuleList()
        self.feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(AttentiveTransformer(self.input_dim, n_a))
            self.feature_transformers.append(FeatureTransformer(self.input_dim, n_d + n_a))

        self.final_mapping = nn.Linear(n_d * (n_steps + 1), num_classes)
        self._init_final()

    def forward(self, x_categ, x_cont):
        x = torch.cat([x_categ.float(), x_cont], dim=1)

        prior = torch.ones_like(x)
        outputs = []
        
        x0 = self.initial_transform(x)
        d = x0[:, :self.n_d]
        a = x0[:, self.n_d:]
        outputs.append(d)

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](a)
            mask = mask * prior
            prior = prior * (self.gamma - mask)

            masked_x = mask * x
            x_step = self.feature_transformers[step](masked_x)

            d = x_step[:, :self.n_d]
            a = x_step[:, self.n_d:]
            outputs.append(d)

        out = torch.cat(outputs, dim=1)
        return self.final_mapping(out)

    def _init_final(self):
        nn.init.kaiming_normal_(self.final_mapping.weight, nonlinearity='relu')
        if self.final_mapping.bias is not None:
            nn.init.zeros_(self.final_mapping.bias)