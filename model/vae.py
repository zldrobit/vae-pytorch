import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latentD):
        super().__init__()

        self.latentD = latentD
        self.enc_conv1 = nn.Conv2d(1, 64, 4, 2, 1) # 14
        self.enc_conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False) # 7
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 1024, 7, bias=False) # 1
        self.enc_bn3 = nn.BatchNorm2d(1024)
        self.enc_mu = nn.Linear(1024, latentD)
        self.enc_logvar = nn.Linear(1024, latentD)

        self.dec_tconv1 = nn.ConvTranspose2d(latentD, 1024, 1, 1, bias=False) # 1
        self.dec_bn1 = nn.BatchNorm2d(1024)
        self.dec_tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False) # 7
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False) # 14
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False) # 32

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = x.view([-1, 1024])
        return torch.distributions.normal.Normal(self.encode_mu(x), F.softplus(self.encode_logvar(x)))

    def decode(self, z):
        z = z.view([-1, 1024, 1, 1])
        x = F.relu(self.dec_bn1(self.dec_tconv1(z)))
        x = F.relu(self.dec_bn2(self.dec_tconv2(z)))
        x = F.relu(self.dec_bn3(self.dec_tconv3(z)))

        img = torch.sigmoid(self.tconv4(x))

        return img

    def forward(self, x):
        q_z = self.encode(x)
        q_z_sample = q_z.rsample()
        x = self.decode(q_z_sample)
        return x, q_z.mean, q_z.scale