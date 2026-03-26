import torch
import torch.nn as nn
import torch.nn.functional as F # Added for F.relu if used in MMD or loss, and for BCE loss in WAE

# --- Model Definitions (adapted from vae/vae.py) ---
class Autoencoder(nn.Module): # Renamed from ConvVAE
    def __init__(self, image_channels=1, h_dim=512, latent_dim=2): # z_dim is now latent_dim
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1) # -> 32x14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # -> 64x7x7
        self.fc_enc = nn.Linear(64 * 7 * 7, h_dim)
        # We still encode mu and logvar to define q(z|x), which is needed for sampling z
        self.fc_mu = nn.Linear(h_dim, latent_dim) # Use latent_dim
        self.fc_logvar = nn.Linear(h_dim, latent_dim) # Use latent_dim
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, h_dim) # Use latent_dim
        self.fc_dec2 = nn.Linear(h_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> 32x14x14
        self.deconv2 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1) # -> 1x28x28
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.conv1(x)); h = self.relu(self.conv2(h))
        h = h.view(h.size(0), -1); h = self.relu(self.fc_enc(h))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h) # Return mu and logvar
        # For uot.py, we directly need the latent code z, not mu and logvar separately for generation.
        # The reparameterize step will effectively give us one sample from q(z|x).
        # If uot.py specifically needs *only* mu for some deterministic encoding, this needs adjustment.
        # For now, let's assume uot.py will use the reparameterized z.
        # So, the encode method used by uot.py should return z.
        return self.reparameterize(mu, logvar) # Return z for compatibility with uot.py expectations

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std # Sample z using reparameterization

    def decode(self, z):
        h = self.relu(self.fc_dec(z)); h = self.relu(self.fc_dec2(h))
        h = h.view(-1, 64, 7, 7); h = self.relu(self.deconv1(h))
        recon_x = self.sigmoid(self.deconv2(h))
        return recon_x # Shape (batch_size, image_channels, H, W)

    # Forward pass for typical VAE/WAE training (returns more info)
    def forward_train(self, x):
        mu, logvar = self.encode_full(x) # a separate method to get mu, logvar
        z = self.reparameterize(mu, logvar) # Sample z
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar # Return z needed for MMD/KL, and recon_x

    # A separate encode_full method if direct mu, logvar is needed by a VAE/WAE trainer
    def encode_full(self, x):
        h = self.relu(self.conv1(x)); h = self.relu(self.conv2(h))
        h = h.view(h.size(0), -1); h = self.relu(self.fc_enc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    # Default forward for uot.py (matches previous Autoencoder simple encode->decode)
    def forward(self, x):
        # For uot.py, it expects encode() to return latent codes directly
        # and then passes these to decode().
        mu, logvar = self.encode_full(x) # Get mu, logvar
        z = self.reparameterize(mu, logvar) # Sample z
        # The original Autoencoder in uot.py did: z = self.encode(x); return self.decode(z)
        # So self.encode(x) should return z.
        # The current self.encode(x) now returns z.
        return self.decode(z) # For compatibility with how uot.py calls AE


class Classifier(nn.Module): # Renamed from SimpleClassifier
    def __init__(self, num_classes=2): # Defaulting to 2 as in SimpleClassifier, uot.py will pass 10
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Ensure input x is (batch_size, 1, H, W) for Conv2d
        if x.dim() == 2: # (batch_size, features) e.g. 784
            if x.shape[1] == 28*28:
                x = x.view(x.size(0), 1, 28, 28) # Reshape flattened to image
            else:
                raise ValueError(f"Input to Classifier has {x.shape[1]} features, expected 28*28 for reshaping.")
        elif x.dim() == 3: # Potentially (batch_size, H, W), add channel dim
             x = x.unsqueeze(1)
        elif x.dim() != 4 or x.shape[1] != 1:
             raise ValueError(f"Input to Classifier must be (B, 1, H, W) or (B, H*W), got {x.shape}")

        x = self.pool1(self.relu(self.conv1(x))); x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1); x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

