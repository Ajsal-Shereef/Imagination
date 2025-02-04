import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# =============================
# 1. Define the VAE Class
# =============================

class VAE(nn.Module):
    def __init__(self, pretrained_model_name, latent_dim=128, num_mixtures=5, mu_p=None, learn_mu_p=False):
        """
        Variational Autoencoder with a pretrained Sentence-BERT encoder, a mixture of Gaussians in the latent space,
        and a decoder to reconstruct Sentence-BERT embeddings.

        Args:
            pretrained_model_name (str): Name of the pretrained Sentence-BERT model.
            latent_dim (int): Dimensionality of the latent space.
            num_mixtures (int): Number of Gaussian components in the mixture.
            mu_p (torch.Tensor or None): Prior means for the Gaussian components. Shape: [num_mixtures, latent_dim]
                                          If None, initializes mu_p as zeros.
            learn_mu_p (bool): If True, allows mu_p to be learnable parameters.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures

        # Encoder: Pretrained Sentence-BERT with fixed weights
        self.encoder = SentenceTransformer(pretrained_model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze encoder weights
        self.embedding_dim = latent_dim
        
        # Linear layers to output mixture parameters
        self.fc_mu = nn.Linear(self.embedding_dim, latent_dim * num_mixtures)
        self.fc_logvar = nn.Linear(self.embedding_dim, latent_dim * num_mixtures)
        self.fc_weights = nn.Linear(self.embedding_dim, num_mixtures)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )
        
        # Apply Xavier initialization to both models
        self.decoder.apply(self.init_weights)
        self.fc_mu.apply(self.init_weights)
        self.fc_logvar.apply(self.init_weights)
        self.fc_weights.apply(self.init_weights)

        # Prior means (mu_p). If not provided, defaults to zero vectors
        if mu_p is None:
            mu_p = torch.zeros(num_mixtures, latent_dim)
        else:
            assert mu_p.shape == (num_mixtures, latent_dim), \
                f"mu_p must have shape ({num_mixtures}, {latent_dim}), but got {mu_p.shape}"
        if learn_mu_p:
            self.mu_p = nn.Parameter(mu_p)  # Make mu_p learnable
        else:
            self.register_buffer('mu_p_buffer', mu_p)  # Register as buffer to move with device
            
    # Function to apply Xavier normal initialization
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        Encode a batch of sentences into mixture parameters.

        Args:
            x (list of str): Batch of input sentences.

        Returns:
            mu (torch.Tensor): Means of the mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            logvar (torch.Tensor): Log variances of the mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            weights (torch.Tensor): Mixture weights. Shape: [batch_size, num_mixtures]
        """
        # Get Sentence-BERT embeddings
        embeddings = self.encoder.encode(x, convert_to_tensor=True, device=self.fc_mu.weight.device)
        # embeddings: [batch_size, embedding_dim]

        # Compute mixture parameters
        mu = self.fc_mu(embeddings).view(-1, self.num_mixtures, self.latent_dim)        # [batch_size, num_mixtures, latent_dim]
        logvar = self.fc_logvar(embeddings).view(-1, self.num_mixtures, self.latent_dim)  # [batch_size, num_mixtures, latent_dim]
        weights = F.softmax(self.fc_weights(embeddings), dim=-1)                       # [batch_size, num_mixtures]

        return mu, logvar, weights

    def reparameterize(self, mu, logvar, weights):
        """
        Reparameterization trick to sample from the mixture of Gaussians.

        Args:
            mu (torch.Tensor): Means of the mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            logvar (torch.Tensor): Log variances of the mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            weights (torch.Tensor): Mixture weights. Shape: [batch_size, num_mixtures]

        Returns:
            z (torch.Tensor): Sampled latent vectors. Shape: [batch_size, latent_dim]
            selected_mu (torch.Tensor): Means of the sampled components. Shape: [batch_size, latent_dim]
            selected_logvar (torch.Tensor): Log variances of the sampled components. Shape: [batch_size, latent_dim]
        """
        batch_size = mu.size(0)
        device = mu.device

        # Sample from each Gaussian component
        eps = torch.randn(batch_size, self.num_mixtures, self.latent_dim).to(device)  # [batch_size, num_mixtures, latent_dim]
        sigma = torch.exp(0.5 * logvar)  # [batch_size, num_mixtures, latent_dim]
        z_i = mu + sigma * eps          # [batch_size, num_mixtures, latent_dim]

        # Compute weighted sum: z = sum(alpha_i * z_i)
        weights = weights.unsqueeze(-1)  # [batch_size, num_mixtures, 1]
        z = torch.sum(weights * z_i, dim=1)  # [batch_size, latent_dim]

        return z, z_i, weights.squeeze(-1)  # Returning z_i for potential debugging

    def decode(self, z):
        """
        Decode latent vectors into reconstructed embeddings.

        Args:
            z (torch.Tensor): Latent vectors. Shape: [batch_size, latent_dim]

        Returns:
            recon_embeddings (torch.Tensor): Reconstructed embeddings. Shape: [batch_size, embedding_dim]
        """
        recon_embeddings = self.decoder(z)  # [batch_size, embedding_dim]
        return recon_embeddings

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (list of str): Batch of input sentences.

        Returns:
            recon (torch.Tensor): Reconstructed embeddings. Shape: [batch_size, embedding_dim]
            mu (torch.Tensor): Means of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            logvar (torch.Tensor): Log variances of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            weights (torch.Tensor): Mixture weights. Shape: [batch_size, num_mixtures]
            z (torch.Tensor): Sampled latent vectors. Shape: [batch_size, latent_dim]
        """
        mu, logvar, weights = self.encode(x)
        z, z_i, weights_expanded = self.reparameterize(mu, logvar, weights)
        recon = self.decode(z)
        return recon, mu, logvar, weights, z

    def kl_divergence(self, mu, logvar, weights):
        """
        Compute the KL divergence between the approximate posterior and the prior.

        Args:
            mu (torch.Tensor): Means of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            logvar (torch.Tensor): Log variances of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            weights (torch.Tensor): Mixture weights. Shape: [batch_size, num_mixtures]

        Returns:
            kl (torch.Tensor): KL divergence for each sample. Shape: [batch_size]
        """
        # If mu_p is learnable, use self.mu_p, else use buffer
        if hasattr(self, 'mu_p'):
            mu_p = self.mu_p  # [num_mixtures, latent_dim]
            mu_p = mu_p.unsqueeze(0)  # [1, num_mixtures, latent_dim]
        else:
            mu_p = self.mu_p_buffer.unsqueeze(0)  # [1, num_mixtures, latent_dim]
            
        # Compute KL divergence for each component and each latent dimension
        # KL(N(mu_i, sigma_i^2) || N(mu_p_i, I)) = 0.5 * (sigma_i^2 + (mu_p_i - mu_i)^2 - 1 - log(sigma_i^2))
        kl = 0.5 * (torch.exp(logvar) + (mu_p - mu)**2 - 1 - logvar)  # [batch, num_mixtures, latent_dim]
        kl = kl.sum(dim=-1)  # Sum over latent dimensions: [batch, num_mixtures]

        # Weight the KL divergence by mixture weights and sum over mixtures
        kl = (weights * kl).sum(dim=1)  # [batch]
        return kl

    def loss_function(self, recon, original, mu, logvar, weights, kl_weight):
        """
        Compute the VAE loss function.

        Args:
            recon (torch.Tensor): Reconstructed embeddings. Shape: [batch_size, embedding_dim]
            original (torch.Tensor): Original embeddings. Shape: [batch_size, embedding_dim]
            mu (torch.Tensor): Means of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            logvar (torch.Tensor): Log variances of mixture components. Shape: [batch_size, num_mixtures, latent_dim]
            weights (torch.Tensor): Mixture weights. Shape: [batch_size, num_mixtures]
            kl_weight (float): Weight for the KL divergence term (for annealing).

        Returns:
            loss (torch.Tensor): Combined VAE loss.
            recon_loss (torch.Tensor): Reconstruction loss.
            kl_loss (torch.Tensor): KL divergence loss.
        """
        # Reconstruction loss: Mean Squared Error
        recon_loss = F.mse_loss(recon, original, reduction='sum')

        # KL divergence loss
        kl = self.kl_divergence(mu, logvar, weights).sum()

        # Total loss with annealed KL weight
        loss = recon_loss + kl_weight * kl
        return loss, recon_loss, kl

    def save(self, path):
        """
        Save the model's state dictionary.

        Args:
            path (str): File path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionary.

        Args:
            path (str): File path from which to load the model.
        """
        self.load_state_dict(torch.load(path, map_location=self.mu_p_buffer.device))
        self.eval()
        print(f"Model loaded from {path}")
