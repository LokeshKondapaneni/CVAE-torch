# MIT License
# Copyright (c) 2024 Lokesh Kondapaneni

import torch
from torch import nn
from typing import List

class cVAE(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 num_classes: int = 40,
                 hidden_layers: List = None,
                 img_size = 64) -> None:
        super(cVAE, self).__init__()
        self.hidden_layers = hidden_layers
        self.img_size = img_size
        
        if self.hidden_layers is None:
            self.hidden_layers = [32,64,128,256,512]
        self.embedding_layer = nn.Linear(num_classes, img_size*img_size)
        self.encoder = self.encoder_model(in_channels= in_channels+1)
        self.fc_mu = nn.Linear(self.hidden_layers[-1]*4, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_layers[-1]*4, latent_dim)
        self.fc_z = nn.Linear(latent_dim + num_classes, 
                              self.hidden_layers[-1]*4)
        self.decoder = self.decoder_model(out_channels=in_channels)
        
        
    def encoder_model(self, in_channels: int = 4):
        encode_layers = []
        for dim in self.hidden_layers:
            module = nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(0.01))
            encode_layers.append(module)
        return nn.Sequential(*encode_layers)
    
    def decoder_model(self, out_channels: int = 3):
        decode_layers = []
        hidden_layers = self.hidden_layers.reverse()
        hidden_layers.append(hidden_layers[-1])
        for i in range(len(hidden_layers)):
            module = nn.Sequential(
                nn.ConvTranspose2d(hidden_layers[i], hidden_layers[i+1],
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_layers[i+1]),
                nn.LeakyReLU(0.01)
            )
            decode_layers.append(module)
        decode_layers.append(nn.Sequential(nn.Conv2d(hidden_layers[-1],
                                               out_channels,
                                               kernel_size=3, padding=1),
                                    nn.Tanh()))
        return nn.Sequential(*decode_layers)
        
    def reparam(self, mu, logvar):
        std_dev = torch.exp(0.5*logvar)
        eps = torch.randn_like(std_dev)
        return mu + std_dev * eps
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        f_mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return f_mu, log_var


    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        z = self.fc_z(z)
        z = z.view(-1, self.hidden_layers[-1], 2, 2)
        z = self.decoder(z)
        return z
    
    def forward(self, x, y):
        embed_y = self.embedding_layer(y)
        embed_y = embed_y.view(-1, self.img_size, self.img_size).unsqueeze(1)
        x = torch.cat([x, embed_y], dim=1)
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        output_img = self.decode(z, y)
        return output_img, mu, logvar