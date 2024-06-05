import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from model.function import *
from model.module import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def main():
    # Parameter settingt
    cnt = 0
    depth = 5
    num_filter = 64
    radars = [60]
    data = 2    # 1 for doppler_data, 2 for range_data
    epochs = 100
    batch_size = [16]
    lr = 0.001
    im_width = 128
    im_height = 128
    inChannel = 3
    loss_fn = nn.MSELoss()

    for i in range(len(radars)):
        for x in range(len(batch_size)):
            radar = radars[i]
            train_hist = []
            dataset, train_sampler, test_sampler = load_data_autoencoder(radar, data)
            dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=train_sampler, shuffle=False, pin_memory = True)
            test_dataloader = DataLoader(dataset, batch_size=batch_size[x],sampler=test_sampler, shuffle=False, pin_memory = True)

            train_loss, val_loss, train_cnt, val_cnt = 0, 0, 0, 0
            model = UNet(3, num_filter)
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model, device_ids=[0, 1])
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            time1 = time()

            for epoch in range(epochs):
                for inputs, _ in dataloader:
                    model.train()
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, inputs)
                    # if(epoch == 99 and cnt == 0):
                    #     ShowPic(inputs, outputs, "train.png")
                    #     cnt += 1
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_cnt += 1
                train_hist.append(train_loss/train_cnt)
                print(f'Epoch {epoch+1} | {radars[i]} GHz, num_filter={num_filter}, train_loss={train_loss/train_cnt}')
                
                if((epoch + 1) % 20 == 0):
                    model.eval()
                    for inputs, _ in test_dataloader:
                        inputs = inputs.to(device)
                        with torch.no_grad():
                            outputs = model(inputs)
                            loss = loss_fn(outputs, inputs)
                            val_loss += loss.item()
                            val_cnt += 1
                        
                    print(f'val_loss={val_loss/val_cnt}')
            
            torch.save(model.state_dict(), '../weights/' + 'autoencoder_range_bilinear_128.pth')
            time2 = time()
            print("Training Time: ", (time2 - time1) / 60)
            
if __name__ == "__main__":
    set_seed()
    main()