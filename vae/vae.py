import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP, Linear
from architectures.Layers import *
from torch.distributions import kl_divergence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
------------------------------------------------------------------------------------
-- InferenceNet class is adapted and modified from Jhosimar George Arias Figueroa
https://github.com/jariasf/GMVAE/tree/master/pytorch
------------------------------------------------------------------------------------
"""

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    """
    This class implements the inference part of the GMVAE.
    """
    super(InferenceNet, self).__init__()

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, num_mixture, temperature=1, hard=0):
    #x = Flatten(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    #Invoking the q(z|x,y)
    mu, logvar, z = self.qzxy(x, y)
    
    y_ = torch.zeros([x.shape[0], num_mixture]).to(device)
    # q(z|x,y)
    zm, zv = [[None] * num_mixture for i in range(2)]
    for i in range(num_mixture):
        y_c = y_ + torch.eye(num_mixture).to(device)[i]
        mu, logvar, _ = self.qzxy(x, y_c)
        zm[i] = mu
        zv[i] = logvar

    output = {'mean': torch.stack(zm, dim=1), 'logvar': torch.stack(zv, dim=1), 'latent': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output

# =============================
# 1. Define the VAE Class
# =============================

class GMVAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden, decoder_hidden, latent_dim=128, num_mixtures=5, mu_p=None):
        """
        Variational Autoencoder with a mixture of Gaussians in the latent space, and a decoder to reconstruct the data.

        Args:
            input_dim (int): Dimention of the data
            encoder_hidden (int): Encoder hidden dimension
            decoder_hidden (int): Decoder hidden dimension
            latent_dim (int): Dimensionality of the latent space.
            num_mixtures (int): Number of Gaussian components in the mixture.
            mu_p (torch.Tensor or None): Prior means for the Gaussian components. Shape: [num_mixtures, latent_dim]
                                          If None, initializes mu_p as zeros.
        """
        super(GMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        self.embedding_dim = latent_dim
        
        #Encoder model
        # Define the encoder model with multiple layers
        self.encoder = MLP(input_dim, latent_dim, encoder_hidden)
        
        #Inference net of the GMVAE
        self.inference_model = InferenceNet(latent_dim, latent_dim, self.num_mixtures)
        
        #Generative net of the VAE
        self.generative_model = MLP(latent_dim, input_dim, decoder_hidden)
        
        # Apply Xavier initialization to both models
        # self.encoder.apply(self.init_weights)
        # self.decoder.apply(self.init_weights)
        # self.fc_mu.apply(self.init_weights)
        # self.fc_logvar.apply(self.init_weights)
        # self.fc_weights.apply(self.init_weights)

        # Prior means (mu_p). If not provided, defaults to zero vectors
        mu_p.shape == (self.num_mixtures, latent_dim), \
                f"mu_p must have shape ({self.num_mixtures}, {latent_dim}), but got {mu_p.shape}"
                
        self.register_buffer('mu_p_buffer', mu_p)  # Register as buffer to move with device
        
        #This is the prior weight of the gaussin, we weight each gaussian equally
        # self.prior_c = torch.full((self.num_mixtures,), 1.0 / self.num_mixtures).to(device)
        self.prior_c = nn.Parameter(torch.zeros(self.num_mixtures))
            
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
            embeddings (torch.Tensor): Embeddigns from the encoder net
        """
        embeddings = self.encoder(x)
        # embeddings: [batch_size, embedding_dim]
        return embeddings

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
            inference_net_out (Dict): Dictionary containing the output of the Inference net
            reconstructed_data (torch.Tensor): The reconstructed data from the decoder [batch_size, Feature_dim]
        """
        encoded_data = self.encode(x)
        inference_net_out = self.inference_model(encoded_data, self.num_mixtures)
        reconstructed_data = self.generative_model(inference_net_out['latent'])
        return inference_net_out, reconstructed_data
    
    def kl_divergence_loss(self, inference_out, captions):
        """
        Compute the KL divergence between the approximate posterior and the prior.

        Args:
            inference_out (torch.Tensor): Dictionary containing all information from inference net of the GMVAE
        Returns:
            kl (torch.Tensor): KL divergence for each sample. Shape: [batch_size]
        """
        if hasattr(self, 'mu_p'):
            mu_p = self.mu_p  # [num_mixtures, latent_dim]
            mu_p = mu_p.unsqueeze(0)  # [1, num_mixtures, latent_dim]
        else:
            mu_p = self.mu_p_buffer.unsqueeze(0)
        # Compute KL divergence for each component and each latent dimension
        kl_divergence = [[0] for i in range(self.num_mixtures)]
        for i in range(self.num_mixtures):
            #KL divergence between two gaussian is given by below formula. The second gaussian variance is identity matrix
            # KL(N(mu_i, sigma_i^2) || N(mu_p_i, I)) = 0.5 * (sigma_i^2 + (mu_p_i - mu_i)^2 - 1 - log(sigma_i^2))
            kl = 0.5 * (torch.exp(inference_out['logvar'][:,i,:]) + (inference_out['mean'][:,i,:] - mu_p[:,i,:])**2 - 1 - inference_out['logvar'][:,i,:])  # [batch, num_mixtures, latent_dim]
            kl = kl.sum(dim=-1)  # Sum over latent dimensions: [batch, num_mixtures]
            kl_divergence[i] = kl
        #The full KL divergence of gaussian mixture model contain two parts
        #E_q(c|x)[KL(q(z|x,c))||p(z|c)] + KL(q(c|x)||p(c))
        kl_z = torch.sum(torch.stack(kl_divergence, dim=-1)*inference_out['prob_cat'], dim=-1)
        
        # #The prior weights are choosen as the cosine similarity between the prior mean and the state captions
        # # Expand prior_means to have the same number of rows as captions
        # prior_means = self.mu_p_buffer.unsqueeze(0).expand(inference_out['mean'].shape[0], -1, -1)  # Shape: [1000, 2, 384]
        # captions = captions.unsqueeze(1)  # Shape: [1000, 1, 384]

        # # Compute cosine similarity along the last dimension
        # cosine_sim = F.cosine_similarity(prior_means, captions, dim=-1)  # Shape: [1000, 2]
        # normalised_cosine_sim = F.softmax(cosine_sim/0.1, dim=-1)
        
        # # Clip the values to avoid log(0) and division by zero
        # prob_cat_clipped = torch.clamp(inference_out['prob_cat'], min=1e-10)
        # normalised_cosine_sim_clipped = torch.clamp(normalised_cosine_sim, min=1e-10)
        
        # kl_c = torch.sum(prob_cat_clipped * torch.log(prob_cat_clipped / normalised_cosine_sim_clipped), dim=-1)
        #Get the prior
        prior = F.softmax(self.prior_c, dim=-1)
        kl_c = torch.sum(inference_out['prob_cat'] * torch.log(inference_out['prob_cat'] / prior), dim=-1)
        return kl_z + kl_c
    
    # def caption_loss(self, inference_out, captions):
    #     class_prob = inference_out['categorical']
    #     # Expand prior_means to have the same number of rows as captions
    #     prior_means = self.mu_p_buffer.unsqueeze(0).expand(class_prob.shape[0], -1, -1)  # Shape: [1000, 2, 384]
    #     captions = captions.unsqueeze(1)  # Shape: [1000, 1, 384]

    #     # Compute cosine similarity along the last dimension
    #     cosine_sim = F.cosine_similarity(prior_means, captions, dim=-1)  # Shape: [1000, 2]
    #     normalised_cosine_sim = F.softmax(cosine_sim, dim=-1)
        
    #     #Compute the KL-divergence
    #     kl_div = normalised_cosine_sim * (torch.log(normalised_cosine_sim + 1e-10) - torch.log(class_prob + 1e-10))
    #     kl_div = torch.sum(kl_div, dim=-1)
    #     # caption_loss = F.kl_div(class_prob, normalised_cosine_sim, reduction = 'none')
    #     return torch.sum(kl_div)

    def loss_function(self, data, caption, inference_out, reconstruction, kl_weight):
        """
        Compute the VAE loss function.

        Args:
            data (torch.Tensor): Original data [batch_size, Feature_dim]
            inference_out (torch.Tensor): Dictionary containing all information from inference net of the GMVAE
            reconstruction (torch.Tensor): The reconstructed data from the decoder [batch_size, Feature_dim]
            kl_weight (float): Weight for the KL divergence term (for annealing).

        Returns:
            loss (torch.Tensor): Combined VAE loss.
            recon_loss (torch.Tensor): Reconstruction loss.
            kl_loss (torch.Tensor): KL divergence loss.
        """
        # Reconstruction loss: Mean Squared Error
        recon_loss = F.mse_loss(reconstruction, data, reduction='sum')

        # KL divergence loss
        kl = torch.sum(self.kl_divergence_loss(inference_out, caption))
        
        #Caption loss
        # caption_loss = self.caption_loss(inference_out, caption)

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
        print(f"[INFO] Model saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionary.

        Args:
            path (str): File path from which to load the model.
        """
        self.load_state_dict(torch.load(path, map_location=self.mu_p_buffer.device))
        self.eval()
        print(f"[INFO] VAE Model loaded from {path}")
