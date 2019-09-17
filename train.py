import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import optim
import torch.nn.functional as F

import os
from tqdm import tqdm
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

dataset = torchvision.datasets.MNIST(params['dataset_dir'], train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=params['batch_size'],
                                         shuffle=True,
                                         drop_last=True)
# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('Training Images {}'.format(params['dataset']))
plt.close('all')

vae = VAE(params['latentD']).to(device)

p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([params['batch_size'], params['latentD']]), requires_grad=False).to(device).float(),
            scale=torch.tensor(np.ones([params['batch_size'], params['latentD']]), requires_grad=False).to(device).float())

optimizer = optim.Adam(vae.parameters(), lr=params['learning_rate'])

if not os.path.exists(params['recon_dir']):
    os.makedirs(params['recon_dir'])
else:
    os.system('rm -f {}/*'.format(params['recon_dir']))

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    with tqdm(total=len(dataset), unit='files') as t:
        for i, (data, _) in enumerate(dataloader, 0):
            data = data.to(device)
            optimizer.zero_grad()
            # KL loss
            imgs, means, scales = vae(data)
            q_z = torch.distributions.normal.Normal(loc=means, scale=scales)
            loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

            # recon loss
            loss_rec = torch.mean(torch.sum(F.binary_cross_entropy(imgs, data, reduction='none'), dim=[2, 3]))
            loss_tot = loss_kl + loss_rec
            loss_tot.backward()
            optimizer.step()
            # print(loss_tot, loss_kl, loss_rec)
            desc = 'loss_tot: {:.3f}, loss_kl: {:.3f}, loss_rec: {:.3f}'.format(
                loss_tot.item(), loss_kl.item(), loss_rec.item()
            )
            t.set_description(desc, refresh=False)
            t.update(params['batch_size'])
            # break
    # output samples


    # break
#
# save model?


