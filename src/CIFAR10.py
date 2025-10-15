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
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import argparse
import sys



import torch
import torchvision
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image



sys.path.append('models_scratch/')
sys.path.append('data/')

from models_scratch import *
from data_utils import *

sns.set(style="whitegrid")

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


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

def filter_cifar10(X, y, minority_class, minority_fraction):
 
    y = y.clone().detach()

    minority_indices = torch.where(y == minority_class)[0]
    majority_indices = torch.where(y != minority_class)[0]

    n1 = len(majority_indices)
    n0 = int(len(minority_indices) * minority_fraction)

    selected_minority_indices = minority_indices[torch.randperm(len(minority_indices))[:n0]]

    final_indices = torch.cat([selected_minority_indices, majority_indices])
    final_indices = final_indices[torch.randperm(len(final_indices))]

    S_filtered = torch.ones(len(y), dtype=torch.long)
    S_filtered[selected_minority_indices] = 0
    S_filtered = S_filtered[final_indices]

    return X[final_indices], y[final_indices], S_filtered



def prepare_cifar10_loaders(batch_size, minority_fraction):
    trainset_raw = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=None)
    testset_raw  = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=None)

    X_train = torch.stack([transforms.ToTensor()(img) for img, _ in trainset_raw])
    y_train = torch.tensor(trainset_raw.targets)

    X_test = torch.stack([transforms.ToTensor()(img) for img, _ in testset_raw])
    y_test = torch.tensor(testset_raw.targets)

    X_train, y_train, S_train = filter_cifar10(X_train, y_train, minority_class=5, minority_fraction=minority_fraction)
    X_test, y_test, S_test = filter_cifar10(X_test, y_test, minority_class=5, minority_fraction=minority_fraction)

    print(f"Train S=0: {(S_train==0).sum().item()} | S=1: {(S_train==1).sum().item()}")
    print(f"Test S=0: {(S_test==0).sum().item()} | S=1: {(S_test==1).sum().item()}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10WithSensitiveAttribute(X_train, S_train, y_train, transform=train_transform)
    testset  = CIFAR10WithSensitiveAttribute(X_test, S_test, y_test, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    testloader  = DataLoader(testset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    return trainloader, testloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def build_model(network, num_classes, input_channels, input_height, input_width, batch_norm, device):
    
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
    elif network == "resnet101":
        net = resnet101(norm_layer=norm_layer, num_classes=num_classes)
    elif network == "resnet152":
        net = resnet152(norm_layer=norm_layer, num_classes=num_classes)    
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


def train_model(network, trainloader, testloader, epochs, batch_size, learning_rate, device, batch_norm, kappa, tau):
    
    model = build_model(network, num_classes=10, input_channels=3, input_height=32, input_width=32, batch_norm=batch_norm, device=device)
    
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_loss_s0, train_loss_s1, train_loss_all = [], [], []
    test_loss_s0, test_loss_s1, test_loss_all = [], [], []
    train_acc_s0, train_acc_s1, train_acc_all = [], [], []
    test_acc_s0, test_acc_s1, test_acc_all = [], [], []
    times, cumulative_time = [], 0
    
    early_stopping_epoch = None
    final_epoch = None

    best_train_loss_s0 = float('inf')
    best_train_loss_s1 = float('inf')
    best_train_loss    = float('inf')
    best_test_loss_s0  = float('inf')
    best_test_loss_s1  = float('inf')
    best_test_loss     = float('inf')
    best_train_acc_s0  = 0
    best_train_acc_s1  = 0
    best_train_acc     = 0
    best_test_acc_s0   = 0
    best_test_acc_s1   = 0
    best_test_acc      = 0
    
    nb_epochs = 0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        total_loss, total_samples = 0.0, 0
        loss_s0_sum, count_s0 = 0.0, 0
        loss_s1_sum, count_s1 = 0.0, 0
        correct_total, correct_s0, correct_s1 = 0, 0, 0
        
        for X_batch, S_batch, y_batch in trainloader:
            X_batch, S_batch, y_batch = X_batch.to(device), S_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
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
        
        avg_loss    = total_loss / total_samples
        avg_loss_s0 = loss_s0_sum / count_s0 
        avg_loss_s1 = loss_s1_sum / count_s1 
        
        acc_total = (correct_total / total_samples) * 100
        acc_s0    = (correct_s0 / count_s0) * 100
        acc_s1    = (correct_s1 / count_s1) * 100 
        
        best_train_loss_s0 = min(best_train_loss_s0, avg_loss_s0) 
        best_train_loss_s1 = min(best_train_loss_s1, avg_loss_s1) 
        best_train_loss    = min(best_train_loss, avg_loss)
        best_train_acc_s0  = max(best_train_acc_s0, acc_s0) 
        best_train_acc_s1  = max(best_train_acc_s1, acc_s1)
        best_train_acc     = max(best_train_acc, acc_total) 
        
        train_loss_s0.append(best_train_loss_s0)
        train_loss_s1.append(best_train_loss_s1)
        train_loss_all.append(best_train_loss)
        train_acc_s0.append(best_train_acc_s0)
        train_acc_s1.append(best_train_acc_s1)
        train_acc_all.append(best_train_acc)
        
        if early_stopping_epoch is None and best_train_acc > tau:
            early_stopping_epoch = epoch + 1
        
        model.eval()
        test_loss_sum, total_test = 0.0, 0
        loss_s0_test_sum, count_s0_test = 0.0, 0
        loss_s1_test_sum, count_s1_test = 0.0, 0
        correct_test_total, correct_s0_test, correct_s1_test = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, S_batch, y_batch in testloader:
                X_batch, S_batch, y_batch = X_batch.to(device), S_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                bsize = y_batch.size(0)
                test_loss_sum += loss.item() * bsize
                total_test += bsize
                
                _, preds = outputs.max(1)
                correct_test_total += preds.eq(y_batch).sum().item()
                
                mask_s0 = (S_batch == 0)
                if mask_s0.any():
                    n_s0 = mask_s0.sum().item()
                    loss_s0 = criterion(outputs[mask_s0], y_batch[mask_s0]).item()
                    loss_s0_test_sum += loss_s0 * n_s0
                    count_s0_test += n_s0
                    correct_s0_test += preds[mask_s0].eq(y_batch[mask_s0]).sum().item()
                
                mask_s1 = (S_batch == 1)
                if mask_s1.any():
                    n_s1 = mask_s1.sum().item()
                    loss_s1 = criterion(outputs[mask_s1], y_batch[mask_s1]).item()
                    loss_s1_test_sum += loss_s1 * n_s1
                    count_s1_test += n_s1
                    correct_s1_test += preds[mask_s1].eq(y_batch[mask_s1]).sum().item()
        
        avg_loss_test    = test_loss_sum / total_test
        avg_loss_s0_test = loss_s0_test_sum / count_s0_test 
        avg_loss_s1_test = loss_s1_test_sum / count_s1_test 
        
        acc_test         = (correct_test_total / total_test) * 100
        acc_s0_test      = (correct_s0_test / count_s0_test) * 100 if count_s0_test > 0 else 0
        acc_s1_test      = (correct_s1_test / count_s1_test) * 100 if count_s1_test > 0 else 0
        
        best_test_loss_s0 = min(best_test_loss_s0, avg_loss_s0_test) 
        best_test_loss_s1 = min(best_test_loss_s1, avg_loss_s1_test) 
        best_test_loss    = min(best_test_loss, avg_loss_test) 
        best_test_acc_s0  = max(best_test_acc_s0, acc_s0_test)
        best_test_acc_s1  = max(best_test_acc_s1, acc_s1_test) 
        best_test_acc     = max(best_test_acc, acc_test)
        
        test_loss_s0.append(best_test_loss_s0)
        test_loss_s1.append(best_test_loss_s1)
        test_loss_all.append(best_test_loss)
        test_acc_s0.append(best_test_acc_s0)
        test_acc_s1.append(best_test_acc_s1)
        test_acc_all.append(best_test_acc)
        
        cumulative_time += time.time() - start_time
        times.append(cumulative_time)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Time: {cumulative_time:.2f}s")
            print(f"  Train -> Loss: S0={avg_loss_s0:.4f}, S1={avg_loss_s1:.4f}, Global={avg_loss:.4f} | "
                  f"Acc: S0={acc_s0:.2f}%, S1={acc_s1:.2f}%, Global={acc_total:.2f}%")
            print(f"  Test  -> Loss: S0={avg_loss_s0_test:.4f}, S1={avg_loss_s1_test:.4f}, Global={avg_loss_test:.4f} | "
                  f"Acc: S0={acc_s0_test:.2f}%, S1={acc_s1_test:.2f}%, Global={acc_test:.2f}%")
        scheduler.step()
        nb_epochs += 1
        
        if best_train_acc_s0 > kappa:
            final_epoch = epoch + 1
            break
            
    if final_epoch is None:
        final_epoch = epochs

    debiasing_duration = (final_epoch - early_stopping_epoch) if early_stopping_epoch is not None else None

    print(f"\nTraining finished in {final_epoch} epochs.")
    if early_stopping_epoch:
        print(f"→ Early stopping threshold (Acc > τ={tau}%) reached at epoch {early_stopping_epoch}.")
    print(f"→ Fairness threshold (Acc_S=0 > κ={kappa}%) reached at epoch {final_epoch}.")
    if early_stopping_epoch:
        print(f"→ Debiasing duration: {debiasing_duration} epochs.")

    return {"times": np.array(times),
    "epoch": np.arange(1, nb_epochs + 1),
    "train_loss_s0": np.array(train_loss_s0),
    "train_loss_s1": np.array(train_loss_s1),
    "train_loss_all": np.array(train_loss_all),
    "test_loss_s0": np.array(test_loss_s0),
    "test_loss_s1": np.array(test_loss_s1),
    "test_loss_all": np.array(test_loss_all),
    "train_acc_s0": np.array(train_acc_s0),
    "train_acc_s1": np.array(train_acc_s1),
    "train_acc_all": np.array(train_acc_all),
    "test_acc_s0": np.array(test_acc_s0),
    "test_acc_s1": np.array(test_acc_s1),
    "test_acc_all": np.array(test_acc_all),
    "early_stopping_epoch": early_stopping_epoch,
    "final_epoch": final_epoch,
    "debiasing_duration": debiasing_duration}

def run_multiple_experiments(network, trainloader, testloader, epochs, batch_size, learning_rate, batch_norm, device, num_runs, kappa, tau):
    records = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        result = train_model(network, trainloader, testloader,
                             epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, batch_norm=batch_norm, device=device, kappa=kappa, tau=tau)
        for i, epoch in enumerate(result["epoch"]):
            record = {
                "num_run": run+1,
                "epoch": result["epoch"][i],
                "time": result["times"][i],
                "train_loss_all": result["train_loss_all"][i],
                "train_loss_s0": result["train_loss_s0"][i],
                "train_loss_s1": result["train_loss_s1"][i],
                "test_loss_all": result["test_loss_all"][i],
                "test_loss_s0": result["test_loss_s0"][i],
                "test_loss_s1": result["test_loss_s1"][i],
                "train_acc_all": result["train_acc_all"][i],
                "train_acc_s0": result["train_acc_s0"][i],
                "train_acc_s1": result["train_acc_s1"][i],
                "test_acc_all": result["test_acc_all"][i],
                "test_acc_s0": result["test_acc_s0"][i],
                "test_acc_s1": result["test_acc_s1"][i],
                "early_stopping_epoch": result["early_stopping_epoch"],
                "final_epoch": result["final_epoch"],
                "debiasing_duration": result["debiasing_duration"],
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    df["epoch"] = df["epoch"].astype(int)  
    
    return df

def run_experiments_for_imbalances(network,imbalance_values, epochs_list, num_runs, batch_size, learning_rate, batch_norm, device, kappa, tau):
    
    df_list = []
    summary_records = []

    for frac, epochs in zip(imbalance_values, epochs_list):
        print(f"\n=== Imbalance: {int(frac*100)}%, {epochs} epochs ===")

        trainloader, testloader = prepare_cifar10_loaders(batch_size=batch_size, minority_fraction=frac)

        for run in range(1, num_runs + 1):
            print(f"\n--- Run {run}/{num_runs} ---")
            result = train_model(network, trainloader, testloader,
                                 epochs=epochs, batch_size=batch_size,
                                 learning_rate=learning_rate, batch_norm=batch_norm, device=device, kappa=kappa, tau=tau)

            df_run = pd.DataFrame({
                "epoch": result["epoch"],
                "time": result["times"],
                "train_loss_all": result["train_loss_all"],
                "train_loss_s0": result["train_loss_s0"],
                "train_loss_s1": result["train_loss_s1"],
                "test_loss_all": result["test_loss_all"],
                "test_loss_s0": result["test_loss_s0"],
                "test_loss_s1": result["test_loss_s1"],
                "train_acc_all": result["train_acc_all"],
                "train_acc_s0": result["train_acc_s0"],
                "train_acc_s1": result["train_acc_s1"],
                "test_acc_all": result["test_acc_all"],
                "test_acc_s0": result["test_acc_s0"],
                "test_acc_s1": result["test_acc_s1"],
            })
            df_run["num_run"] = run
            df_run["imbalance"] = int(frac * 100)
            df_list.append(df_run)

            early_stopping_epoch = result.get("early_stopping_epoch")
            final_epoch = result.get("final_epoch")
            debiasing_duration = result.get("debiasing_duration")
            if early_stopping_epoch and early_stopping_epoch > 0:
                debiasing_ratio = (debiasing_duration / early_stopping_epoch) * 100
            else:
                debiasing_ratio = 0

            summary_records.append({
                "network": network,
                "imbalance": int(frac * 100),
                "run": run,
                "early_stopping_epoch": early_stopping_epoch,
                "final_epoch": final_epoch,
                "debiasing_duration": debiasing_duration,
                "debiasing_ratio": debiasing_ratio
            })

    df_all = pd.concat(df_list, ignore_index=True)
    
    df_summary = pd.DataFrame(summary_records)
    summary_path = f"results/CIFAR-10/adam/overcost/{network}_debiasing_summary_kappa_{kappa}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Debiasing summary saved to {summary_path}")
    
    all_path = f"results/CIFAR-10/adam/epochs/{network}_per_epoch_results_kappa_{kappa}.csv"
    df_all.to_csv(all_path, index=False)
    print(f"Per-epoch training results saved to {all_path}")

    return df_all


def plot_metrics_by_imbalance(df, network, kappa, show_test=False, show_loss=False):
    n_epochs = int(df["epoch"].max())
    pdf_path = f"results/CIFAR-10/adam/figures/{network}_metrics_by_imbalance_kappa_{kappa}_{'with_test' if show_test else 'train_only'}.pdf"

    df_loss_train = df.melt(id_vars=["num_run", "epoch", "time", "imbalance"],
                            value_vars=["train_loss_all", "train_loss_s0", "train_loss_s1"],
                            var_name="group", value_name="train_loss")
    df_acc_train = df.melt(id_vars=["num_run", "epoch", "time", "imbalance"],
                           value_vars=["train_acc_all", "train_acc_s0", "train_acc_s1"],
                           var_name="group", value_name="train_acc")

    df_loss_test, df_acc_test = None, None
    if show_test:
        df_loss_test = df.melt(id_vars=["num_run", "epoch", "time", "imbalance"],
                               value_vars=["test_loss_all", "test_loss_s0", "test_loss_s1"],
                               var_name="group", value_name="test_loss")
        df_acc_test = df.melt(id_vars=["num_run", "epoch", "time", "imbalance"],
                              value_vars=["test_acc_all", "test_acc_s0", "test_acc_s1"],
                              var_name="group", value_name="test_acc")

    mapping = {
        "train_loss_all": "Global", "train_loss_s0": "A=0", "train_loss_s1": "A=1",
        "test_loss_all": "Global", "test_loss_s0": "A=0", "test_loss_s1": "A=1",
        "train_acc_all": "Global", "train_acc_s0": "A=0", "train_acc_s1": "A=1",
        "test_acc_all": "Global", "test_acc_s0": "A=0", "test_acc_s1": "A=1"
    }

    df_loss_train["group"] = df_loss_train["group"].map(mapping)
    df_acc_train["group"] = df_acc_train["group"].map(mapping)
    if show_test:
        df_loss_test["group"] = df_loss_test["group"].map(mapping)
        df_acc_test["group"] = df_acc_test["group"].map(mapping)

    palette = {"A=0": "blue", "A=1": "orange", "Global": "green"}
    imbalance_levels = sorted(df["imbalance"].unique())
    n_cols = len(imbalance_levels)
    n_rows = 4 if show_test and show_loss else 2 if show_loss or show_test else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.75 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    def plot_metric(ax, data, y_value, log_scale=False):
        sns.lineplot(data=data, x="epoch", y=y_value, hue="group",
                     estimator="mean", errorbar="sd", palette=palette, ax=ax, legend=False)
        if log_scale:
            ax.set_yscale('log')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for col_idx, imbalance in enumerate(imbalance_levels):
        row_idx = 0
        if show_loss:
            plot_metric(axes[row_idx, col_idx], df_loss_train[df_loss_train["imbalance"] == imbalance], "train_loss", log_scale=True)
            axes[row_idx, col_idx].set_title(rf"$\zeta = {imbalance}\%$", fontsize=14)
            axes[row_idx, 0].set_ylabel("Train loss", fontsize=14)
            row_idx += 1
        else:
            axes[row_idx, col_idx].set_title(rf"$\zeta = {imbalance}\%$", fontsize=14)

        plot_metric(axes[row_idx, col_idx], df_acc_train[df_acc_train["imbalance"] == imbalance], "train_acc")
        axes[row_idx, col_idx].axhline(kappa, color='red', linestyle='--', linewidth=1)
        if col_idx == 0:
            axes[row_idx, 0].set_ylabel("Train accuracy (%)", fontsize=14)
        row_idx += 1

        if show_test:
            if show_loss:
                plot_metric(axes[row_idx, col_idx], df_loss_test[df_loss_test["imbalance"] == imbalance], "test_loss", log_scale=True)
                axes[row_idx, 0].set_ylabel("Test loss", fontsize=12)
                row_idx += 1

            plot_metric(axes[row_idx, col_idx], df_acc_test[df_acc_test["imbalance"] == imbalance], "test_acc")
            axes[row_idx, 0].set_ylabel("Test accuracy (%)", fontsize=12)
            axes[row_idx, col_idx].set_xlabel("Epoch")

    for row in range(n_rows):
        for col in range(1, n_cols):
            axes[row, col].set_ylabel("")
        for col in range(n_cols):
            if row < (n_rows - 1):
                axes[row, col].set_xlabel("")

    group_legend = [Line2D([0], [0], color=color, lw=2, label=label)
                    for label, color in palette.items()]
    red_line = Line2D([0], [0], color='red', lw=1, linestyle='--', label=r"$\kappa$ threshold")

    # Légende en bas
    fig.legend(group_legend + [red_line], [*palette.keys(), r"$\kappa$ threshold"],
               loc="lower center", ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.05))

    # Ajustement de l'espace pour laisser place à la légende en bas
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--tau', type=float, default=90.0)
    parser.add_argument('--kappa', type=float, default=90.0)
    parser.add_argument('--network', type=str, default="resnet18")
    parser.add_argument('--batch_norm', type=boolean_string, default="True")
    parser.add_argument('--imbalance_values', type=float, nargs='+', default=[0.01])
    parser.add_argument('--epochs_list', type=int, nargs='+', default=[5000])
    parser.add_argument('--show_test', type=boolean_string, default="False")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    df_results = run_experiments_for_imbalances(args.network,
                                            imbalance_values=args.imbalance_values,
                                            num_runs=args.num_runs, epochs_list=args.epochs_list, batch_size=args.batch_size,
                                            learning_rate=args.lr, batch_norm=args.batch_norm, device=device, kappa=args.kappa, tau=args.tau)
    
    plot_metrics_by_imbalance(df_results, args.network, args.kappa, show_test=True, show_loss=True)
    plot_metrics_by_imbalance(df_results, args.network, args.kappa, show_test=False, show_loss=False)
    

if __name__ == "__main__":
    main()