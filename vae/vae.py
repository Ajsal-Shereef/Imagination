import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================
# 1. Define the VAE Class
# =============================

class VAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden, decoder_hidden, latent_dim=128, num_mixtures=5, mu_p=None, learn_mu_p=False):
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
        if mu_p is None:
            self.num_mixtures = 1
        self.embedding_dim = latent_dim
        
        #Encoder model
        # Define the encoder model with multiple layers
        self.encoder = MLP(input_dim, latent_dim, encoder_hidden)
        
        # Linear layers to output mixture parameters
        self.fc_mu = nn.Linear(self.embedding_dim, latent_dim * self.num_mixtures)
        self.fc_logvar = nn.Linear(self.embedding_dim, latent_dim * self.num_mixtures)
        if mu_p is not None:
            self.fc_weights = nn.Linear(self.embedding_dim, self.num_mixtures)

        # Decoder
        self.decoder = MLP(latent_dim, input_dim, decoder_hidden)
        
        # Apply Xavier initialization to both models
        # self.encoder.apply(self.init_weights)
        # self.decoder.apply(self.init_weights)
        # self.fc_mu.apply(self.init_weights)
        # self.fc_logvar.apply(self.init_weights)
        # self.fc_weights.apply(self.init_weights)

        # Prior means (mu_p). If not provided, defaults to zero vectors
        if mu_p is None:
            mu_p = torch.zeros(self.num_mixtures, latent_dim)
        else:
            assert mu_p.shape == (self.num_mixtures, latent_dim), \
                f"mu_p must have shape ({self.num_mixtures}, {latent_dim}), but got {mu_p.shape}"
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
        embeddings = self.encoder(x)
        # embeddings: [batch_size, embedding_dim]

        # Compute mixture parameters
        if not self.num_mixtures==1:
            mu = self.fc_mu(embeddings).view(-1, self.num_mixtures, self.latent_dim)        # [batch_size, num_mixtures, latent_dim]
            logvar = self.fc_logvar(embeddings).view(-1, self.num_mixtures, self.latent_dim)  # [batch_size, num_mixtures, latent_dim]
            weights = F.softmax(self.fc_weights(embeddings), dim=-1)                       # [batch_size, num_mixtures]
            return mu, logvar, weights
        else:
            mu = self.fc_mu(embeddings)
            logvar = self.fc_logvar(embeddings)
            return mu, logvar, []

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
        
        if not self.num_mixtures==1:
            # Sample from each Gaussian component
            eps = torch.randn(batch_size, self.num_mixtures, self.latent_dim).to(device)  # [batch_size, num_mixtures, latent_dim]
            sigma = torch.exp(0.5 * logvar)  # [batch_size, num_mixtures, latent_dim]
            z_i = mu + sigma * eps          # [batch_size, num_mixtures, latent_dim]

            # Compute weighted sum: z = sum(alpha_i * z_i)
            weights = weights.unsqueeze(-1)  # [batch_size, num_mixtures, 1]
            z = torch.sum(weights * z_i, dim=1)  # [batch_size, latent_dim]

            return z, z_i, weights.squeeze(-1)  # Returning z_i for potential debugging
        else:
            # Sample from each Gaussian component
            eps = torch.randn(batch_size, self.latent_dim).to(device)  # [batch_size, latent_dim]
            sigma = torch.exp(0.5 * logvar)  # [batch_size, latent_dim]
            z = mu + sigma * eps          # [batch_size, latent_dim]
            return z

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
        if not self.num_mixtures==1:
            mu, logvar, weights = self.encode(x)
            z, z_i, weights_expanded = self.reparameterize(mu, logvar, weights)
        else:
            mu, logvar, weights = self.encode(x)
            z = self.reparameterize(mu, logvar, weights)
            
        recon = self.decode(z)
        return recon, mu, logvar, weights, z
    
    def log_prob_mixture_gaussian(self, x, alpha, mu, sigma):
        """
        Compute the log probability of a mixture of Gaussians.
        x: [batch_size, num_components, num_samples, feature_dim]
        alpha: [batch_size, num_components] mixture weights
        mu: [batch_size, num_components, feature_dim] means
        sigma: [batch_size, num_components, feature_dim] standard deviations
        """
        batch_size, num_components, num_samples, feature_dim = x.shape

        # Constants for normalization
        log_2pi = torch.log(torch.tensor(2 * torch.pi)).to(device)
        constant = 0.5 * feature_dim * log_2pi

        # Clamping sigma for numerical stability
        sigma = torch.clamp(sigma, min=1e-6)

        # Compute the quadratic term
        diff = x - mu.unsqueeze(2)  # [batch_size, num_components, num_samples, feature_dim]
        quad_term = torch.sum((diff ** 2) / (sigma.unsqueeze(2) ** 2), dim=-1)  # Sum over feature_dim

        # Compute the log probability
        log_prob_components = (
            torch.log(alpha).unsqueeze(2)  # Log mixture weights
            - constant  # Normalization constant
            - 0.5 * quad_term  # Quadratic term
            - torch.sum(torch.log(sigma), dim=-1).unsqueeze(2)  # Log std deviation
        )

        # Use log-sum-exp trick for stability
        log_prob_mixture = torch.logsumexp(log_prob_components, dim=1)  # Sum over num_components
        return torch.sum(log_prob_mixture, dim=-1)  # Sum over num_samples


    def kl_divergence_approximation(self, alpha_p, mu_p, logsigma_p, num_samples=500):
        """
        Approximate the KL divergence between p(x) and g(x) using Monte Carlo sampling.

        Parameters:
        - alpha_p: Mixture weights for p(x).
        - mu_p: Means of the Gaussian components for p(x).
        - logsigma_p: Log of Standard deviations of the Gaussian components for p(x).
        - mu_g: Means of the Gaussian components for g(x) (fixed sigma = 1).
        - num_samples: Number of Monte Carlo samples to use for the approximation.

        Returns:
        - kl_divergence: Approximate KL divergence.
        """
        sigma_p = torch.exp(logsigma_p)
        batch_size, num_components, feature_dim = mu_p.shape
        
        # Sample from p(x)
        z_samples_each_data = mu_p.unsqueeze(2) + sigma_p.unsqueeze(2) * torch.randn(batch_size, num_components, num_samples, feature_dim).to(device)

        # Compute log-probabilities under p(x)
        log_p_x = self.log_prob_mixture_gaussian(z_samples_each_data, alpha_p, mu_p, sigma_p)
        
        if hasattr(self, 'mu_p'):
            mu_g = self.mu_p  # [num_mixtures, latent_dim]
            mu_g = mu_p.unsqueeze(0)  # [1, num_mixtures, latent_dim]
        else:
            mu_g = self.mu_p_buffer.unsqueeze(0).repeat(z_samples_each_data.shape[0], 1, 1)
                
        # Compute log-probabilities under g(x)
        # Here g(x) is a mixture of Gaussians with unit variance and uniform weights (1/M)
        alpha_g = (torch.ones(z_samples_each_data.shape[0], z_samples_each_data.shape[1]) / z_samples_each_data.shape[1]).to(device)
        sigma_g = torch.ones_like(sigma_p)  # All standard deviations are fixed at 1 for g(x)

        log_g_x = self.log_prob_mixture_gaussian(z_samples_each_data, alpha_g, mu_g, sigma_g)

        # Compute KL divergence as expectation of log p(x) - log g(x) over samples from p(x)
        kl_divergence = log_p_x - log_g_x

        return kl_divergence

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
            if not self.num_mixtures==1:
                mu_p = self.mu_p_buffer.unsqueeze(0)  # [1, num_mixtures, latent_dim]
            else:
                mu_p = self.mu_p_buffer

        # Compute KL divergence for each component and each latent dimension
        # KL(N(mu_i, sigma_i^2) || N(mu_p_i, I)) = 0.5 * (sigma_i^2 + (mu_p_i - mu_i)^2 - 1 - log(sigma_i^2))
        kl = 0.5 * (torch.exp(logvar) + (mu_p - mu)**2 - 1 - logvar)  # [batch, num_mixtures, latent_dim]
        kl = kl.sum(dim=-1)  # Sum over latent dimensions: [batch, num_mixtures]

        # Weight the KL divergence by mixture weights and sum over mixtures
        if not self.num_mixtures==1:
            kl = (1/self.num_mixtures * kl).sum(dim=1)  # [batch]
        return kl
    
    def weight_regularisation_loss(self, weight):
        """
        Forces VAE to equally weight the given gaussians
        
        Args:
            weight (torch.Tensor): Weight learned from the VAE model
            
        Returns:
            loss (torch.Tensor): Weight regularisation loss
        """
        return -torch.sum(weight * torch.log(weight + 1e-9))

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
            weight_regualarisation_loss (torch.Tensor): Weight regularisation loss
        """
        # Reconstruction loss: Mean Squared Error
        recon_loss = F.mse_loss(recon, original, reduction='mean')

        # KL divergence loss
        kl = torch.mean(self.kl_divergence_approximation(weights, mu, logvar))
        
        #Weight regularisation term
        # weight_regualarisation_loss = self.weight_regularisation_loss(weights)

        # Total loss with annealed KL weight
        loss = recon_loss + kl_weight * kl# - weight_regualarisation_loss
        return loss, recon_loss, kl#, weight_regualarisation_loss

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
