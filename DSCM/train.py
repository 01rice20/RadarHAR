import os
import random
import h5py
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from torchvision.datasets import ImageFolder
from model.function import *
from model.module import *
import argparse
from thop import profile
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.init(project="inno_project")
print('Using device:', device)

def train(radar=60):
    cnt = 0
    epochs = 100
    batch_size = [8, 16]
    dense_1 = [64, 64, 128, 128, 256, 256]
    dense_2 = [32, 64, 64, 128, 128, 256]
    learn_rate = [2e-4, 1e-4]
    drop=0.1
    num_filter = 64
    num_class = 5
    acc_hist = []
    patience = 3
    result = {}
    bestscore = best_score()
    criterion = nn.CrossEntropyLoss()
    premodel1 = []
    premodel2 = []
    premodel1 = UNet(3, 64)
    premodel2 = UNet(3, 64)

    # if torch.cuda.device_count() > 1:
    #     premodel1 = nn.DataParallel(premodel1, device_ids=[0, 1]).to(device)
    #     premodel2 = nn.DataParallel(premodel2, device_ids=[0, 1]).to(device)

    premodel1, premodel2, dataset, train_sampler, val_sampler, test_sampler, nj = load_data_weight(premodel1, premodel2, radar)
    premodel1 = premodel1.to(device)
    premodel2 = premodel2.to(device)

    for x in range(len(batch_size)):
        dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=train_sampler, shuffle=False, pin_memory = True)
        val_dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=val_sampler, shuffle=False, pin_memory = True)
        test_dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=test_sampler, shuffle=False, pin_memory = True)

        for k in range(len(dense_1)):
            for i in range(len(learn_rate)):
                model = []
                early_stopping = EarlyStopping(patience)
                model = Classificate(premodel1, premodel2, num_class, dense_1[k], dense_2[k], drop).to(device)
                total = sum([param.nelement() for param in model.parameters()])
                print('Number of parameter: % .4fM' % (total / 1e6))
                optimizer = optim.Adam(model.parameters(), lr=learn_rate[i], weight_decay=1e-6)
                train_loss, train_cnt, val_loss, val_cnt, accuracy = 0, 0, 0, 0, 0

                for epoch in range(epochs):
                    model.train()
                    for doppler, ranges, labels in dataloader:
                        doppler = doppler.to(device)
                        ranges = ranges.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(doppler, ranges)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    
                    print(f'Epoch {epoch + 1}, {radar} GHz, batch_size={batch_size[x]}, '
                            f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[i]}, '
                            f'train_loss={train_loss/train_cnt}')
                    wandb.log({"train_loss": train_loss/train_cnt})
                    
                    if((epoch + 1) % 5 == 0):
                        model.eval()
                        v_pred, v_true = [], []
                        istest = False
                        for doppler, ranges, labels in val_dataloader:
                            with torch.no_grad():
                                doppler = doppler.to(device)
                                ranges = ranges.to(device)
                                labels = labels.to(device)
                                outputs = model(doppler, ranges)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                                val_cnt += 1
                                _, predicted = torch.max(outputs, 1)
                                v_pred.append(predicted)
                                v_true.append(labels)

                        v_pred = torch.cat(v_pred, dim=0).cpu().numpy()
                        v_true = torch.cat(v_true, dim=0).cpu().numpy()
                        accuracy = score(v_pred, v_true, istest)
                        vloss_hist = val_loss/val_cnt
                        print(f'val_loss={vloss_hist}, val_acc={round(accuracy, 5)}')
                        wandb.log({"val_loss": vloss_hist, "val_accuracy": accuracy})
                        if early_stopping(vloss_hist):
                            print("Early stopping")
                            break

                model.eval()
                t_pred, t_true = [], []
                istest = True
                isfinal = False
                for doppler, ranges, labels in test_dataloader:
                    doppler = doppler.to(device)
                    ranges = ranges.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs = model(doppler, ranges)
                        _, predicted = torch.max(outputs, 1)
                        t_pred.append(predicted)
                        t_true.append(labels)

                t_pred = torch.cat(t_pred, dim=0).cpu().numpy()
                t_true = torch.cat(t_true, dim=0).cpu().numpy()
                cm, precision, recall, f1, accuracy = score(t_pred, t_true, istest)
                bestscore(cm, precision, recall, f1, accuracy, isfinal)
                wandb.log({"accuracy": accuracy})
                print(f'test_precision={round(precision, 5)}, test_recall={round(recall, 5)}, '
                        f'test_f1={round(f1, 5)}, test_accuracy={round(accuracy, 5)}')
                torch.save(model.state_dict(), f'../weights/{radar}_{batch_size[x]}_{dense_1[k]}_{dense_2[k]}_{learn_rate[i]}.pth')
    
    isfinal = True
    result = bestscore(cm, precision, recall, f1, accuracy, isfinal)
    wandb.log({"Best Precision": result['precision'], "Best Recall": result['recall'], 
                "Best F1": result['f1'], "Best Recall": result['recall'], "Best accuracy": result['accuracy']})
                
if __name__ == "__main__":
    
    set_seed()
    train()