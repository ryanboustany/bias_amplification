import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math
import copy
import io
import time
import matplotlib.pyplot as plt
import argparse
from scipy.stats import t


import torch
import torchvision
import torch
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
from torch.nn.utils import parameters_to_vector


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")

import sys
sys.path.append('models_scratch/')
sys.path.append('data/')
from models_scratch import *
from data_utils import *

sns.set(style="whitegrid")

## Double precision or not?
doublePrecision = False


if doublePrecision:
    torch.set_default_dtype(torch.float64)


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

class CIFAR10WithSensitiveAttribute(Dataset):
    def __init__(self, X, S, y, transform=None):
        self.X = X
        self.S = S
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        target = self.y[idx]
        sensitive = self.S[idx]

        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)

        return img, sensitive, target

def get_cifar10_train_loader(batch_size=256, size=32):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=False, transform=transform)
    return trainset

def get_cifar10_test_loader(size=32):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=False, transform=transform)
    return testset 
 

def filter_cifar10(X, y, minority_class=1, majority_class=9, minority_fraction=0.1):
    # Clone pour éviter d'altérer y original
    y = y.clone().detach()

    # Sélection des indices des classes minoritaire et majoritaire
    minority_indices = torch.where(y == minority_class)[0]
    majority_indices = torch.where(y == majority_class)[0]

    n1 = len(majority_indices)
    n0 = int((minority_fraction / (1 - minority_fraction)) * n1)

    # Sous-échantillonnage aléatoire des minoritaires
    selected_minority_indices = minority_indices[torch.randperm(len(minority_indices))[:n0]]

    # Regrouper et mélanger les indices
    final_indices = torch.cat([selected_minority_indices, majority_indices])
    final_indices = final_indices[torch.randperm(len(final_indices))]

    # Filtrage des données et labels
    X_filtered = X[final_indices]
    y_filtered = y[final_indices]

    # Remapping des labels pour la classification binaire : minority_class → 0, majority_class → 1
    y_filtered = (y_filtered == majority_class).long()

    # Variable sensible : 0 pour minoritaire, 1 pour majoritaire
    S_filtered = y_filtered.clone()

    return X_filtered, y_filtered, S_filtered


def build_model(network, num_classes, input_channels, input_height, input_width, batch_norm = True, device='cuda'):
    
    if batch_norm:
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = None

    if network == "vgg11":
        net = VGG("VGG11", num_classes=num_classes, batch_norm=batch_norm)
    elif network == "vgg19":
        net = VGG("VGG19", num_classes=num_classes, batch_norm=batch_norm)
    elif network == "resnet18":
        net = resnet18(norm_layer=norm_layer, num_classes=num_classes)
    elif network == "resnet34":
        net = resnet34(norm_layer=norm_layer, num_classes=num_classes)
    elif network == "resnet50":
        net = resnet50(norm_layer=norm_layer, num_classes=num_classes)
    elif network == "densenet121":
        net = densenet121(norm_layer=norm_layer, num_classes=num_classes,
                          input_channels=input_channels, input_height=input_height, input_width=input_width)
    elif network == "mobilenet":
        net = MobileNet(num_classes=num_classes,
                          input_channels=input_channels, input_height=input_height, input_width=input_width)
    elif network == "squeezenet":
        net = SqueezeNet(num_classes=num_classes,
                          input_channels=input_channels, input_height=input_height, input_width=input_width)
    elif network == "lenet":
        net = LeNet5(num_classes=num_classes, input_channels=input_channels,
                     input_height=input_height, input_width=input_width)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)
    
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters in {network}: {num_params:,}")
    
    return net

def train_S1_phase(model, optimizer, criterion, dataloader, device, epochs=5000, epsilon=1e-2):
    metrics = {"epoch": [], "loss": [], "acc": [], "grad_norm": []}
    model.train()
    best_acc = 0

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        count = 0

        optimizer.zero_grad()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        for X_batch, S_batch, y_batch in dataloader:
            mask = (S_batch == 1)
            if mask.sum().item() == 0:
                continue
            X_s1 = X_batch[mask].to(device)
            y_s1 = y_batch[mask].to(device)

            outputs = model(X_s1)
            loss = criterion(outputs, y_s1)
            loss.backward()

            running_loss += loss.item() * X_s1.size(0)
            count += X_s1.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y_s1).sum().item()

        with torch.no_grad():
            grad_vector = parameters_to_vector(
                [p.grad for p in model.parameters() if p.grad is not None]
            )
            grad_norm = grad_vector.norm().item()

        optimizer.step()

        avg_loss = running_loss / count if count > 0 else 0
        avg_acc = 100 * correct / count if count > 0 else 0
        best_acc = max(best_acc, avg_acc)

        metrics["epoch"].append(epoch)
        metrics["loss"].append(avg_loss)
        metrics["acc"].append(best_acc)
        metrics["grad_norm"].append(grad_norm)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%, GradNorm: {grad_norm:.6f}")

        # Stop if gradient norm is too small
        if grad_norm < epsilon:
            print(f"→ Early stopping: ‖∇L₁‖/n₁ < {epsilon} reached at epoch {epoch+1}")
            break

    theta1 = parameters_to_vector(model.parameters()).detach().clone()
    return metrics, theta1

def compute_theta_distances(model, theta1):
    current_vector = parameters_to_vector(model.parameters())
    diff = current_vector - theta1

    l2_norm = torch.norm(diff, p=2).item()
    sup_norm = torch.norm(diff, p=float('inf')).item()
    relative_norm = l2_norm / torch.norm(current_vector, p=2).item()

    return l2_norm, sup_norm, relative_norm

def train_full_phase(model, optimizer, criterion, dataloader, device, epochs, theta1,
                     stagnation_epochs=10, min_increase_ratio=0.001):
    metrics = {
        "epoch": [],
        "loss_s0": [],
        "loss_s1": [],
        "loss_global": [],
        "acc_s0": [],
        "acc_s1": [],
        "acc_global": [],
        "l2_norm": [],
        "sup_norm": [],
        "relative_norm": [],
    }

    model.train()
    T_final = None

    previous_l2 = None
    stagnation_counter = 0

    for epoch in range(epochs):
        model.zero_grad()
        total_loss, total_samples = 0.0, 0
        loss_s0_sum, count_s0 = 0.0, 0
        loss_s1_sum, count_s1 = 0.0, 0
        correct_total, correct_s0, correct_s1 = 0, 0, 0

        for X_batch, S_batch, y_batch in dataloader:
            X_batch, S_batch, y_batch = X_batch.to(device), S_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            bsize = y_batch.size(0)
            total_loss += loss.item() * bsize
            total_samples += bsize

            _, preds = outputs.max(1)
            correct_total += preds.eq(y_batch).sum().item()

            mask_s0 = (S_batch == 0)
            if mask_s0.any():
                n_s0 = mask_s0.sum().item()
                loss_s0 = criterion(outputs[mask_s0], y_batch[mask_s0]).item()
                loss_s0_sum += loss_s0 * n_s0
                count_s0 += n_s0
                correct_s0 += preds[mask_s0].eq(y_batch[mask_s0]).sum().item()

            mask_s1 = (S_batch == 1)
            if mask_s1.any():
                n_s1 = mask_s1.sum().item()
                loss_s1 = criterion(outputs[mask_s1], y_batch[mask_s1]).item()
                loss_s1_sum += loss_s1 * n_s1
                count_s1 += n_s1
                correct_s1 += preds[mask_s1].eq(y_batch[mask_s1]).sum().item()

        # Gradient norm
        with torch.no_grad():
            grad_vector = parameters_to_vector([p.grad for p in model.parameters() if p.grad is not None]) 
            grad_norm = grad_vector.norm().item()

        optimizer.step()

        # Compute distances to theta1
        l2_norm, sup_norm, relative_norm = compute_theta_distances(model, theta1)

        # Stagnation detection
        if previous_l2 is not None:
            if l2_norm <= (1 + min_increase_ratio) * previous_l2:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        previous_l2 = l2_norm

        if stagnation_counter >= stagnation_epochs:
            print(f"→ Early stop: ‖θ - θ₁‖ did not increase by more than {min_increase_ratio*100:.1f}% "
                  f"for {stagnation_epochs} consecutive epochs")
            break

        # Metrics
        avg_loss = total_loss / total_samples
        avg_loss_s0 = loss_s0_sum / count_s0 if count_s0 > 0 else 0
        avg_loss_s1 = loss_s1_sum / count_s1 if count_s1 > 0 else 0
        acc_total = (correct_total / total_samples) * 100
        acc_s0 = (correct_s0 / count_s0) * 100 if count_s0 > 0 else 0
        acc_s1 = (correct_s1 / count_s1) * 100 if count_s1 > 0 else 0

        metrics["epoch"].append(epoch)
        metrics["loss_s0"].append(avg_loss_s0)
        metrics["loss_s1"].append(avg_loss_s1)
        metrics["loss_global"].append(avg_loss)
        metrics["acc_s0"].append(acc_s0)
        metrics["acc_s1"].append(acc_s1)
        metrics["acc_global"].append(acc_total)
        metrics["l2_norm"].append(l2_norm)
        metrics["sup_norm"].append(sup_norm)
        metrics["relative_norm"].append(relative_norm)


        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Loss S0={avg_loss_s0:.4f}, Acc S0={acc_s0:.2f}%, "
                  f"Loss S1={avg_loss_s1:.4f}, Acc S1={acc_s1:.2f}%, "
                  f"Global Loss={avg_loss:.4f}, Global Acc={acc_total:.2f}%, "
                  f"‖θ - θ₁‖={l2_norm:.4f}, ‖.‖∞={sup_norm:.4f}, rel={relative_norm:.4f}, "
                  f"‖∇L‖ = {grad_norm:.4f}")

    metrics["T_final"] = epoch + 1

    print(f"\nTraining finished in {epoch+1} epochs.")

    return metrics


def run_multiple_experiments(network, device,
                             n_runs=3, epochs_phase1=5000, epochs_phase2=1000,
                             learning_rate=1e-2, epsilon=1e-2):
    import pandas as pd
    import torch
    import time
    from torch.utils.data import DataLoader, TensorDataset
    from torch import optim, nn
    from torch.nn.utils import parameters_to_vector

    detailed_records = []
    summary_records = []

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        trainset, testset = get_cifar10_train_loader(), get_cifar10_test_loader()
        X_train = torch.stack([img for img, _ in trainset])
        y_train = torch.tensor(trainset.targets)
        X_test = torch.stack([img for img, _ in testset])
        y_test = torch.tensor(testset.targets)

        X_train, y_train, S_train = filter_cifar10(X_train, y_train, minority_fraction=0.03)
        X_test, y_test, S_test = filter_cifar10(X_test, y_test, minority_fraction=0.03)

        train_dataset = TensorDataset(X_train, S_train, y_train)
        trainloader = DataLoader(train_dataset, batch_size=len(y_train), shuffle=True)

        model = build_model(network, 2, input_channels=3, input_height=32, input_width=32, device=device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        start_time = time.time()
        metrics_phase1, theta1 = train_S1_phase(model, optimizer, criterion, trainloader, device,
                                                epochs=epochs_phase1, epsilon=epsilon)
        metrics_phase2 = train_full_phase(model, optimizer, criterion, trainloader, device,
                                          epochs=epochs_phase2, theta1=theta1,
                                          stagnation_epochs=10, min_increase_ratio=0.001)
        end_time = time.time()

        T_final = metrics_phase2.get("T_final")
        last_l2 = metrics_phase2["l2_norm"][-1]
        last_sup = metrics_phase2["sup_norm"][-1]
        last_rel = metrics_phase2["relative_norm"][-1]

        for i, epoch in enumerate(metrics_phase2["epoch"]):
            detailed_records.append({
                "network": network,
                "run": run + 1,
                "epoch": epoch,
                "train_loss_all": metrics_phase2["loss_global"][i],
                "train_loss_s0": metrics_phase2["loss_s0"][i],
                "train_loss_s1": metrics_phase2["loss_s1"][i],
                "train_acc_all": metrics_phase2["acc_global"][i],
                "train_acc_s0": metrics_phase2["acc_s0"][i],
                "train_acc_s1": metrics_phase2["acc_s1"][i],
                "theta_l2": metrics_phase2["l2_norm"][i],
                "theta_sup": metrics_phase2["sup_norm"][i],
                "theta_relative": metrics_phase2["relative_norm"][i],
                "T_final": T_final,
                "elapsed_time": end_time - start_time
            })

        summary_records.append({
            "network": network,
            "run": run + 1,
            "T_final": T_final,
            "theta_l2": last_l2,
            "theta_sup": last_sup,
            "theta_relative": last_rel,
            "elapsed_time": end_time - start_time
        })

    df_detailed = pd.DataFrame(detailed_records)
    df_summary = pd.DataFrame(summary_records)

    # Stats descriptives
    stats = df_summary[["T_final", "theta_l2", "theta_sup", "theta_relative", "elapsed_time"]].agg(
        ["mean", "std", "min", "max"]
    ).reset_index().rename(columns={"index": "stat"})

    # Sauvegarde
    detailed_path = f"results/CIFAR-2/{network}_debiasing_detailed.csv"
    summary_path = f"results/CIFAR-2/{network}_debiasing_summary.csv"
    stats_path = f"results/CIFAR-2/{network}_debiasing_stats.csv"

    df_detailed.to_csv(detailed_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"\nSaved detailed records to: {detailed_path}")
    print(f"Saved summary records to: {summary_path}")
    print(f"Saved aggregate stats to: {stats_path}")

    return df_detailed, df_summary, stats



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--network', type=str, default="resnet18")
    parser.add_argument('--epochs_phase1', type=int,  default=5000)
    parser.add_argument('--epochs_phase2', type=int, default=5000)
    parser.add_argument('--epsilon', type=float, default=1e-2)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    df_detailed, df_summary, stats = run_multiple_experiments(args.network, device=device, n_runs=args.num_runs, epochs_phase1=args.epochs_phase1, epochs_phase2=args.epochs_phase2, learning_rate=args.lr, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
