import torch
import torch.nn as nn
import torchvision

def conv_block(in_f, out_f, *args, **kwargs):
    return(nn.Sequential(
        nn.Conv3d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm3d(out_f),
        nn.ReLU(),
        nn.Dropout()
    ))

def unconv_block(in_f, out_f, *args, **kwargs):
    return(nn.Sequential(
        nn.ConvTranspose3d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm3d(out_f),
        nn.ReLU()
    ))

class MyUNet(nn.Module):
    def __init__(self, h_dim=40, z_dim=4):
        super(MyUNet, self).__init__()

        self.fc1 = nn.Linear(8**3 * 12, h_dim) # FC internal enc
        self.fc2 = nn.Linear(h_dim, z_dim)  # FC MU
        self.fc3 = nn.Linear(h_dim, z_dim)  # FC STD
        self.fc4 = nn.Linear(z_dim, h_dim)  # FC internal dec
        self.fc5 = nn.Linear(h_dim, 8**3 * 12) # FC internal dec

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(2) # Luego se le pasan los indices
        # TODO: revisar si conviene poner ConvTranspose3D mejor.

        self.conv_block1 = conv_block(1, 4, kernel_size=3, padding=1,
                                      stride=1)
        self.conv_block2 = conv_block(4, 8, kernel_size=3, padding=1,
                                      stride=1)
        self.conv_block3 = conv_block(8, 12, kernel_size=3, padding=1,
                                      stride=1)

        self.unconv_block1 = unconv_block(12, 8, kernel_size=3, padding=1,
                                         stride=1)
        self.unconv_block2 = unconv_block(8, 4, kernel_size=3, padding=1,
                                         stride=1)
        self.unconv_block3 = unconv_block(4, 1, kernel_size=3, padding=1,
                                         stride=1)
        lay_en = [
            nn.Conv3d(1, 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(4),
            nn.MaxPool3d(2),
        ]
        lay_de = []

        self.encoder = nn.Sequential(*lay_en)
        self.decoder = nn.Sequential(*lay_de)

    def encode(self, x):
        """ Encoding of the input """
        # 32 x 32 x 32 x 1
        x = self.conv_block1(x)
        x, indices1 = self.pool(x)

        # 16 x 16 x 16 x 4
        x = self.conv_block2(x)
        x, indices2 = self.pool(x)

        # 8 x 8 x 8 x 8
        x = self.conv_block3(x)
        # x, indices3 = self.pool(x)

        # 8 x 8 x 8 x 12
        h = x.view(-1, 8**3 * 12)
        h = self.relu(self.fc1(h))
        return self.fc2(h), self.fc3(h), indices1, indices2

    def reparameterize(self, mu, log_var):
        """ Reparameterizing trick done by the VAE """
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, indices1, indices2):
        """
        Decoding of the latent code made by the encoder and reparameterized
        """
        h = self.relu(self.fc5(self.fc4(z)))
        h = h.view(-1, 12, 8, 8, 8)

        # 8 x 8 x 8 x 12
        h = self.unconv_block1(h)

        # 8 x 8 x 8 x 8
        h = self.unpool(h, indices2)
        h = self.unconv_block2(h)

        # 16 x 16 x 16 x 4
        h = self.unpool(h, indices1)
        h = self.unconv_block3(h)

        # 32 x 32 x 32 x 1
        m = nn.Sigmoid()
        return m(h)

    def forward(self, x):
        """
        FeedForward Network
        """
        mu, log_var, indices1, indices2 = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z, indices1, indices2)
        return x_reconst, mu, log_var, z

