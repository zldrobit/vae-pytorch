import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import optim

import parser
import numpy as np
from matplotlib import pyplot as plt
import time
from config import params
from model import VAE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

dataset = torchvision.datasets.MNIST('/home/fangjiacong/Dataset/MNIST', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=params['batch_size'],
                                         shuffle=True)
# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('Training Images {}'.format(params['dataset']))
plt.close('all')

vae = VAE(params['latentD']).to(device)

optimizer = optim.Adam(vae.parameters(), lr=params['learning_rate'])

zero_grad()
backward()

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(dataloader, 0):
        optim optimize minimize

    output samples

save model?


