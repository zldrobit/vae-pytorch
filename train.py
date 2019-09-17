import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
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

test_dataset = torchvision.datasets.MNIST(params['dataset_dir'], train=False, transform=transform, download=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=params['valid_size'],
                                         shuffle=True,
                                         drop_last=True)
# Plot the training images.
sample_batch = next(iter(dataloader))
sample_batch = sample_batch[0].to(device)[:100]
plt.figure(figsize=(10, 10))
plt.axis("off")
sample_batch_grid = vutils.make_grid(sample_batch, nrow=10, padding=2, normalize=True)
plt.imshow(np.transpose(sample_batch_grid.cpu(), (1, 2, 0)))
plt.savefig('Training Images {}'.format(params['dataset']))
plt.close('all')

test_batch = next(iter(test_dataloader))
test_batch = test_batch[0].to(device)[:100]
test_batch_grid = vutils.make_grid(test_batch, nrow=10, padding=2, normalize=True)

vae = VAE(params['latentD']).to(device)

p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([params['batch_size'], params['latentD']]), requires_grad=False).to(device).float(),
            scale=torch.tensor(np.ones([params['batch_size'], params['latentD']]), requires_grad=False).to(device).float())

optimizer = optim.Adam(vae.parameters(), lr=params['learning_rate'])

if not os.path.exists(params['recon_dir']):
    os.makedirs(params['recon_dir'])
else:
    os.system('rm -f {}/*'.format(params['recon_dir']))

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
# print(sample_batch.shape)
writer.add_graph(vae, sample_batch)
writer.add_image('Train/Original', sample_batch_grid, 0)
writer.add_image('Test/Original', test_batch_grid, 0)

rand_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([params['valid_size'], params['latentD']]), requires_grad=False).to(device).float(),
            scale=torch.tensor(np.ones([params['valid_size'], params['latentD']]), requires_grad=False).to(device).float()).sample()

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    vae.train()
    with tqdm(total=len(dataset), unit='files') as t:
        for i, (data, _) in enumerate(dataloader, 0):
            step = i + epoch * len(dataloader)
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
            writer.add_scalar('Loss/KL', loss_kl, step)
            writer.add_scalar('Loss/Recon', loss_rec, step)
            writer.add_scalar('Loss/Total', loss_tot, step)
        
            # break
    # output reconstruction
    vae.eval()
    imgs, *_ = vae(sample_batch)
    sample_batch_recon = vutils.make_grid(imgs, nrow=10, padding=2, normalize=True)
    writer.add_image('Train/Recon', sample_batch_recon, step)
    imgs, *_ = vae(test_batch)
    test_batch_recon = vutils.make_grid(imgs, nrow=10, padding=2, normalize=True)
    writer.add_image('Test/Recon', test_batch_recon, step)
    
    # output random samples
    imgs = vae.decode(rand_z)
    rand_samples = vutils.make_grid(imgs, nrow=10, padding=2, normalize=True)
    writer.add_image('Random Z', rand_samples, step)

    # break
#
# save model?


