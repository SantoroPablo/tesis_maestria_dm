import torch.nn as nn
import torch
import torchvision

# Clase que arma el autoencoder
class VAE(nn.Module):
    """
    Variational Auto Encoder architechture
    """
    def __init__(self, h_dim=40, z_dim=4):
        super(VAE, self).__init__()
        # Capas como listas
        lay_en = [
            nn.Conv3d(1, 16, kernel_size=2, stride=2), #16x16x16x16            
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1), #8x16x16x16
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Conv3d(8, 8, kernel_size=2, stride=2), #8x8x8x8
            nn.Dropout()
        ]
        lay_de = [
            nn.ConvTranspose3d(8, 12, kernel_size=2, stride=2),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2)
        ]
        # Ordenando las capas secuencialmente
        self.encoder = nn.Sequential(*lay_en)
        self.decoder = nn.Sequential(*lay_de)
        # 8x5x5
        self.fc1 = nn.Linear(8**4, h_dim) # FC internal enc
        self.fc2 = nn.Linear(h_dim, z_dim)  # FC MU
        self.fc3 = nn.Linear(h_dim, z_dim)  # FC STD
        self.fc4 = nn.Linear(z_dim, h_dim)  # FC internal dec
        self.fc5 = nn.Linear(h_dim, 8**4) # FC internal dec
    def encode(self, x):
        """ Encoding of the input """
        m = nn.ReLU()
        h = self.encoder(x)
        h = h.view(-1, 8**4)
        h = m(self.fc1(h))
        return self.fc2(h), self.fc3(h)
    def reparameterize(self, mu, log_var):
        """ Reparameterizing trick done by the VAE """
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        """
        Decoding of the latent code made by the encoder and reparameterized
        """
        m = nn.ReLU()
        h = m(self.fc5(self.fc4(z)))
        h = h.view(-1, 8, 8, 8, 8)
        h = self.decoder(h)
        m = nn.Sigmoid()
        return m(h)
    def forward(self, x):
        """
        FeedForward Network
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var, z

