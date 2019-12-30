import os

os.system("pip install tqdm")

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

TRAINING_DIR = "/storage/faces/lfw-deepfunneled/lfw-deepfunneled"

###################### CREATE THE DATASET AND DATALOADER ###########################################

"""
 [!] NOTE : The mean and std of the data was computed beforehand.
"""
mean = np.load("mean.npy")
std = np.load("std.npy")

trans = transforms.Compose([
  transforms.Resize(100),
  transforms.ToTensor(),
  transforms.Normalize(mean=mean, std=std)
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, shuffle=True, batch_size=16, num_workers=12)

###################### CREATE THE VAE MODEL ##########################################################

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], 84*84*64)  

class UnFlatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], 64, 84, 84)

class FaceVAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(8, 16, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.ReLU(True),
      Flatten(),
      nn.Linear(64*84*84, 400),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(400, 20)
    self.logvar_layer = nn.Linear(400, 20)
    self.decoder = nn.Sequential(
      nn.Linear(20, 400),
      nn.ReLU(True),
      nn.Linear(400, 64*84*84),
      nn.ReLU(True),
      UnFlatten(),
      nn.ConvTranspose2d(64, 32, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 3, kernel_size=5)
    )

  def reparam_(self, mu, logvar):
    std = torch.exp(logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return mu, logvar

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparam_(mu, logvar)
    return self.decode(z), mu, logvar

########################### CREATE THE LOSS AND OPTIMIZER ##############################

def loss_function(x_pred, x, mu, logvar):
  mse = fn.mse_loss(x_pred, x)
  kl = -5e-4 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return mse + kl

vae = FaceVAE().cuda()
sgd = optim.Adam(vae.parameters(), lr=1e-3)

########################### TRAIN THE MODEL ############################################

for epoch in range(25):
  start_time = time.time()

  for x, _ in tqdm(trainloader):
    x = x.cuda().float()
    
    sgd.zero_grad()

    x_pred, mu, logvar = vae.forward(x)
    loss = loss_function(x_pred, x, mu, logvar)

    loss.backward()
    sgd.step()

  end_time = time.time()

  print("\n")
  print("[{}] Loss={}".format(epoch+1, loss.detach().cpu().numpy()))
  print("Elapsed Time={}".format(end_time - start_time))
  print("\n")

  if epoch+1 % 5 == 0:
    torch.save(vae.state_dict(), "/artifacts/faces-vae-epoch-{}-weights.pth".format(epoch+1))

torch.save(vae.state_dict(), "/artifacts/faces-vae-weights.pth")
