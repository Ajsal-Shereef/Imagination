import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from geomloss import SamplesLoss
from torch.nn.utils import spectral_norm
from architectures.mlp import MLP, Linear
from architectures.cnn import CNNLayer, CNN
from architectures.m2_vae.stochastic import GaussianSample
from architectures.m2_vae.vae import VariationalAutoencoder
from helper_functions.utils import sample_gumbel_softmax, logsumexp_stable
# from architectures.m2_vae.distributions import log_standard_categorical
from architectures.m2_vae.vae import FeatureEncoder, Decoder, LadderEncoder, LadderDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [h_dim, y_dim, hidden] = dims
        # self.mlp = MLP(x_dim, y_dim, hidden, dropout_prob=0.20)
        self.mu_c = Linear(h_dim, y_dim)
        self.log_var_c = Linear(h_dim, y_dim)
        # self.dense0 = nn.Linear(x_dim, 256)
        # self.dense1 = nn.Linear(256, 512)
        # self.logits = nn.Linear(512, y_dim)

    def forward(self, x):
        mu_c = self.mu_c(x)
        log_var_c = self.log_var_c(x)
        return mu_c, log_var_c
    
# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 5 * 5, 1)),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, x):
        return self.net(x)
    
    def discriminator_loss(self, real_data, fake_data):
        real_scores = self.net(real_data)
        fake_scores = self.net(fake_data.detach())
        real_loss = self.criterion(real_scores.squeeze(), torch.ones_like(real_scores.squeeze()))
        fake_loss = self.criterion(fake_scores.squeeze(), torch.zeros_like(fake_scores.squeeze()))
        gp = self.gradient_penalty(real_data, fake_data.detach())
        return real_loss + fake_loss + 0.10*gp
    
    def gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        d_interpolated = self(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty
    
class MineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MineNetwork, self).__init__()
        self.net = MLP(input_size, 1, hidden_size, hidden_activation=nn.ELU())
        
    def forward(self, x, y):
        # Concatenate the inputs
        inputs = torch.cat((x, y), dim=1)
        return self.net(inputs)
    
    def mine_estimate(self, T_joint, T_marginal):
        # Compute the loss using the Donsker-Varadhan representation
        # T_joint: Output of the network for joint samples (paired)
        # T_marginal: Output for marginal samples (unpaired)
        joint_term = torch.mean(T_joint)
        marginal_term = torch.mean(torch.exp(T_marginal))
        loss = (joint_term - torch.log(marginal_term + 1e-6) + 1e-6)
        return loss


class DeepGenerativeModel(nn.Module):
    def __init__(self, dims, label_weight, recon_weight):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [self.x_dim, self.y_dim, self.h_dim, self.z_dim, self.classifier_hidden, encoder_hidden_layers] = dims
        self.label_weight = label_weight
        self.recon_weight = recon_weight
        super(DeepGenerativeModel, self).__init__()
        self.encoder = FeatureEncoder([self.x_dim, self.h_dim, encoder_hidden_layers])
        self.z_latent = GaussianSample(self.h_dim, self.z_dim)
        self.decoder = Decoder([self.z_dim, self.y_dim, self.x_dim])
        # self.classifier = GaussianSample(self.h_dim, self.z_dim)
        self.classifier = MLP(self.h_dim, self.y_dim, [64])
        # self.z_projection = Linear(self.z_dim, 32)
        # self.c_projection = Linear(self.y_dim, 32)
        
        # self.mi_loss_discriminator = MLP(self.z_dim + self.y_dim, 1, [16])

        # self.discriminator = Discriminator(self.x_dim)
        # self.discriminator_criterion = nn.BCEWithLogitsLoss(reduction='sum')

        self.register_buffer('c_prior_mu', torch.full((self.z_dim,), 1.0 / self.z_dim))
        self.register_buffer('c_prior_logvar', torch.zeros(self.z_dim))  # Log variance of 0
        
        self.reconstruction_loss_function = nn.BCELoss(reduction='none')
        self.label_loss_function = nn.CrossEntropyLoss()


    def forward(self, x):
        # Add label and data and generate latent variable
        h = self.encoder(x)
        z_latent, z_mu, z_log_var = self.z_latent(h)
        c_logits = self.classify(h) 
        c = sample_gumbel_softmax(c_logits, self.training)
        # Reconstruct data point from latent data and label
        # if y is not None:
        #     x_reconstructed = self.decoder(z_latent, y)
        # else:
        if self.training:
            x_reconstructed = self.decoder(z_latent, c)
        else:
            x_reconstructed = self.decoder(z_mu, c)

        return x_reconstructed, z_latent, z_mu, z_log_var, c_logits, c
    
    
    def classify(self, h):
        c_logits = self.classifier(h)
        return c_logits
    
    def generate(self, x_input: torch.Tensor, c_cond: torch.Tensor, use_mean_z: bool = True) -> torch.Tensor:
        """
        Generates images conditionally based on the input image and the provided soft label

        Args:
            x_input: Input images (B, 3, 40, 40)
            c_cond: Conditioning soft labels (B, num_classes)
            use_mean_z: If True, uses the mean latent vector mu; otherwise, samples z

        Returns:
            x_generated: Generated images (B, 3, 40, 40)
        """
        h = self.encoder(x_input)
        z, mu, logvar = self.z_latent(h)
        if use_mean_z:
            z = mu  # Use mean of q(z|x)
        x_generated = self.decoder(z, c_cond)
        return x_generated

    # def sample(self, z, y):
    #     """
    #     Samples from the Decoder to generate an x.
    #     :param z: latent normal variable
    #     :param y: label (one-hot encoded)
    #     :return: x
    #     """
    #     y = y.float()
    #     x = self.decoder(torch.cat([z, y], dim=1))
    #     return x
    
    # def decoder_loss(self, fake_scores):
    #     return self.discriminator_criterion(fake_scores, torch.ones_like(fake_scores))
    
    # def mi_loss(self, z_samples, c_samples, temperature=0.07):
    #     # Projection heads
    #     z_proj = self.z_projection(z_samples)  # [batch_size, projection_dim]
    #     c_proj = self.c_projection(c_samples)  # [batch_size, projection_dim]

    #     # Normalize projections
    #     z_proj = F.normalize(z_proj, dim=1)
    #     c_proj = F.normalize(c_proj, dim=1)

    #     # Compute similarity scores (dot product)
    #     scores = torch.matmul(z_proj, c_proj.t()) / temperature  # [batch_size, batch_size]

    #     # Positive pairs are on the diagonal
    #     pos_mask = torch.eye(z_samples.size(0), device=z_samples.device)

    #     # Separate logits into positive and negative
    #     pos_logits = scores * pos_mask  # Diagonal values are the positive logits
    #     neg_logits = scores * (1 - pos_mask)  # Off-diagonal values are the negatives

    #     # Compute the loss
    #     loss = torch.logsumexp(neg_logits, dim=1) - torch.logsumexp(pos_logits, dim=1)
    #     return loss.mean()

    
    # def mi_loss(self, z_samples, c_samples):
    #     # Joint samples
    #     z_joint = z_samples  # [batch_size, latent_dim_z]
    #     c_joint = c_samples  # [batch_size, latent_dim_c]
    #     # Marginal samples (shuffled c)
    #     c_shuffled = c_samples[torch.randperm(z_joint.shape[0])]  # Shuffle c in the batch
    #     c_marginal = c_shuffled
    #     # Compute scores for joint samples
    #     joint_latent = torch.cat([z_joint, c_joint], dim=-1)
    #     s_joint = self.mi_loss_discriminator(joint_latent)  # [batch_size, 1]
    #     # Compute scores for marginal samples
    #     marginal_latent = torch.cat([z_joint, c_marginal], dim=-1)
    #     s_marginal = self.mi_loss_discriminator(marginal_latent)  # [batch_size, 1]

    #     # Donsker-Varadhan representation of MI
    #     joint_loss = s_joint.mean()  # Expectation over joint samples
    #     marginal_loss = torch.logsumexp(s_marginal, dim=0) - torch.log(torch.tensor(c_joint.size(0), dtype=torch.float)+ 1e-8)

    #     # Mutual Information Loss
    #     mi_loss = (joint_loss - marginal_loss)
    #     return mi_loss
    
    def kl_divergence_z(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def kl_divergence_c(self, logits):
        q_c = F.softmax(logits, dim=-1)                 # [batch_size, num_classes]
        log_q_c = F.log_softmax(logits, dim=-1)         # [batch_size, num_classes]
        K = logits.size(-1)                             # Number of classes
        # Compute log uniform prior probability
        log_uniform_prob = -torch.log(torch.tensor(K, dtype=torch.float, device=logits.device))
        kl_per_sample = torch.sum(q_c * (log_q_c - log_uniform_prob), dim=1)    # [batch_size]
        kl = kl_per_sample.mean()
        return kl
    
    def L(self, x, x_recon, mu_z, logvar_z, y, logits_c, x_c, ws_weight, kl_weight):
        #Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='none')
        recon_loss = torch.mean(recon_loss.sum(dim=-1))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)
        label_loss = self.label_loss_function(logits_c, y)
        
        #Cycle consistency loss
        _, _, z_mu, _, c_logits, _ = self.forward(x_recon)
        x_reconstructed_label_loss = self.label_loss_function(c_logits, y)
        x_reconstructed_z_loss = F.mse_loss(z_mu, mu_z)
        cycle_loss = x_reconstructed_z_loss + x_reconstructed_label_loss

        #Creating some auxiliary losses
        # Generate random labels
        labels = F.one_hot(torch.randint(0, self.y_dim, (x_recon.shape[0],)), num_classes=self.y_dim)
        generated_images = self.generate(x, labels.to(device).float())
        _, _, z_mu, _, gen_logits, _ = self.forward(generated_images)
        aux_reconstruction_loss = F.mse_loss(z_mu, mu_z)
        aux_classification_loss = self.label_loss_function(gen_logits, labels.to(device).float())
        aux_loss = aux_reconstruction_loss + aux_classification_loss
        
        # Mutual information loss to disentagle z and c
        # mi_loss = self.mi_loss(mu_z, x_c)
        # mi_loss = 0

        # Wasserstein loss
        wasserstein_loss = self.sliced_wasserstein_distance(x, x_recon)
        # wasserstein_loss = 0
        
        total_loss = self.recon_weight*recon_loss + kl_weight*kl_z + 0.01*kl_c + self.label_weight*label_loss + cycle_loss + ws_weight*wasserstein_loss + aux_loss
        return total_loss, label_loss
    
    def U(self, x, x_recon, mu_z, logvar_z, logits_c, x_c, ws_weight, kl_weight):
        #Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='none')
        recon_loss = torch.mean(recon_loss.sum(dim=-1))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)
        
        #Self supervised label loss
        _, _, z_mu, _, recon_logit, _ = self.forward(x_recon)
        reconstructed_label_loss = self.label_loss_function(recon_logit, torch.argmax(logits_c, dim=-1))
        x_reconstructed_z_loss = F.mse_loss(z_mu, mu_z)
        cycle_loss = x_reconstructed_z_loss + reconstructed_label_loss
        
        # Mutual information loss to disentagle z and c
        # mi_loss = self.mi_loss(mu_z, x_c)
        # mi_loss = 0

        # Wasserstein loss
        wasserstein_loss = self.sliced_wasserstein_distance(x, x_recon)
        # wasserstein_loss = 0

        #Creating some auxiliary losses
        # Generate random labels
        labels = F.one_hot(torch.randint(0, self.y_dim, (x_recon.shape[0],)), num_classes=self.y_dim)
        generated_images = self.generate(x, labels.to(device).float())
        _, _, z_mu, _, gen_logits, _ = self.forward(generated_images)
        aux_reconstruction_loss = F.mse_loss(z_mu, mu_z)
        aux_classification_loss = self.label_loss_function(gen_logits, labels.to(device).float())
        aux_loss = aux_reconstruction_loss + aux_classification_loss
        
        total_loss = self.recon_weight*recon_loss + kl_weight*kl_z + 0.01*kl_c + cycle_loss + ws_weight*wasserstein_loss + aux_loss
        return total_loss, recon_loss, kl_z, kl_c, wasserstein_loss
    
    def sliced_wasserstein_distance(self, real_samples, generated_samples, num_projections=100, device='cuda'):
        # Flatten the samples
        real_flat = real_samples.view(real_samples.size(0), -1)
        gen_flat = generated_samples.view(generated_samples.size(0), -1)

        eps = 1e-7  # Small constant to prevent division by zero

        # Generate random projections
        projections = torch.randn(num_projections, real_flat.size(1), device=device)
        projections = F.normalize(projections, dim=1)  # Normalize to have unit length

        # Project the samples onto the random directions
        real_projections = real_flat @ projections.t()
        gen_projections = gen_flat @ projections.t()

        # Sort the projected samples
        real_projections, _ = torch.sort(real_projections, dim=0)
        gen_projections, _ = torch.sort(gen_projections, dim=0)

        # Compute the SWD
        swd = torch.mean((real_projections - gen_projections) ** 2)

        return swd

    
    # def sinkhorn_loss(self, real_samples, generated_samples):
    #     # Flatten and normalize samples
    #     real_flat = real_samples.view(real_samples.size(0), -1)
    #     gen_flat = generated_samples.view(generated_samples.size(0), -1)

    #     # Optionally, normalize the data
    #     real_flat = (real_flat - real_flat.mean(dim=1, keepdim=True)) / (real_flat.std(dim=1, keepdim=True) + 1e-8)
    #     gen_flat = (gen_flat - gen_flat.mean(dim=1, keepdim=True)) / (gen_flat.std(dim=1, keepdim=True) + 1e-8)

    #     # Use SamplesLoss with appropriate parameters
    #     sinkhorn_loss_fn = SamplesLoss(
    #         loss='sinkhorn',
    #         p=2,
    #         blur=0.05,         # Adjust blur (epsilon ** 0.5)
    #         scaling=0.9,
    #         debias=False       # Set to False if you encounter issues with debiasing
    #     )

    #     sinkhorn_dist = sinkhorn_loss_fn(real_flat, gen_flat)

    #     return sinkhorn_dist

    
    # #Loss unction for labelled data
    # def L(self, x, x_recon, mu_z, logvar_z, mu_c, logvar_c, y, sigma2=1.0, sigma_y2=1.0, kl_weight=1):
    #     """
    #     Computes the loss function for GMVAE

    #     Args:
    #         x: Input images (B, 3, 40, 40)
    #         mu: Mean of q(z|x) (B, latent_dim)
    #         logvar: Log variance of q(z|x) (B, latent_dim)
    #         z: Sampled latent variables (B, latent_dim)
    #         y: Soft labels (B, num_classes), representing p(y|x)
    #     Returns:
    #         total_loss: The total loss for the batch
    #         loss_dict: Dictionary containing individual loss components
    #     """
    #     batch_size = x.size(0)
    #     device = x.device

    #      # Reconstruction loss (MSE loss)
    #     recon_loss = F.mse_loss(x_recon, x, reduction='mean') / (2 * sigma2)

    #     # KL divergence for z
    #     kl_z = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    #     # KL divergence for c
    #     prior_mu_c = self.c_prior_mu.to(device)
    #     prior_logvar_c = self.c_prior_logvar.to(device)
    #     kl_c = -0.5 * torch.mean(
    #         1 + logvar_c - prior_logvar_c - ((mu_c - prior_mu_c).pow(2) + logvar_c.exp()) / prior_logvar_c.exp()
    #     )

    #     # Total VAE loss
    #     # Total loss
    #     total_loss = recon_loss + kl_weight*kl_z + kl_weight*kl_c
    #     # Compute E_q(c|x)[|| y - c ||^2] = || y - mu_c ||^2 + Tr(Sigma_c)
    #     diff = y - mu_c
    #     sq_diff = diff.pow(2).sum(dim=1)  # (B,)
    #     trace_sigma_c = torch.exp(logvar_c).sum(dim=1)  # (B,)
    #     expected_sq_diff = sq_diff + trace_sigma_c  # (B,)
    #     scalled_class_loss = (expected_sq_diff/(2 * sigma_y2)).mean()
    #     # Add label loss to total loss
    #     total_loss += scalled_class_loss
    #     return total_loss, scalled_class_loss

    # # Loss function for unlabeled data
    # def U(self,x, x_recon, mu_z, logvar_z, mu_c, logvar_c, sigma2=1.0,  kl_weight=1):
    #     """
    #     Computes the loss function for GMVAE

    #     Args:
    #         x: Input images (B, 3, 40, 40)
    #         mu: Mean of q(z|x) (B, latent_dim)
    #         logvar: Log variance of q(z|x) (B, latent_dim)
    #         z: Sampled latent variables (B, latent_dim)
    #         q_c_probs: Class probabilities q(c|x) (B, num_classes)
    #         y: Soft labels (B, num_classes), representing p(y|x)
    #     Returns:
    #         total_loss: The total loss for the batch
    #         loss_dict: Dictionary containing individual loss components
    #     """
    #     batch_size = x.size(0)
    #     device = x.device

    #      # Reconstruction loss (MSE loss)
    #     recon_loss = F.mse_loss(x_recon, x, reduction='mean') / (2 * sigma2)

    #     # KL divergence for z
    #     kl_z = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    #     # KL divergence for c
    #     prior_mu_c = self.c_prior_mu.to(device)
    #     prior_logvar_c = self.c_prior_logvar.to(device)
    #     kl_c = -0.5 * torch.mean(
    #         1 + logvar_c - prior_logvar_c - ((mu_c - prior_mu_c).pow(2) + logvar_c.exp()) / prior_logvar_c.exp()
    #     )

    #     # Total VAE loss
    #     # Total loss
    #     total_loss = recon_loss + kl_weight*kl_z + kl_weight*kl_c
    #     return total_loss, recon_loss, kl_z, kl_c

        
    def save(self, save_path, epochs=0):
        torch.save(self.state_dict(), f'{save_path}/model.pt')
        print(f"[INFO] model saved after {epochs} at, {save_path}/model.pt")
        with open(f'{save_path}/specs.json', 'w') as f:
            f.write('''{
            "input_dim": %d,
            "num_goals": %d,
            "h_dim": %d,
            "latent_dim": %d
            }''' % (self.x_dim, self.y_dim, self.h_dim, self.z_dim))
            
    def load(self, model_dir):
        params = torch.load(model_dir, map_location=device)
        self.load_state_dict(params)
            


class StackedDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, features):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim])

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x):
        _, x, _ = self.features.encoder(x)
        logits = self.classifier(x)
        return logits

class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # Generative p(x|z,y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(LadderDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder([e.in_features + y_dim, e.out_features, e.z_dim])

        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classifier([x_dim, h_dim[0], y_dim])

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0]+y_dim, h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder)-1:
                x, (z, mu, log_var) = encoder(torch.cat([x, y], dim=1))
            else:
                x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            # If at top, encoder == decoder,
            # use prior for KL.
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += self._kld(z, (l_mu, l_log_var))

            # Perform downword merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += self._kld(*kl)

        x_mu = self.reconstruction(torch.cat([z, y], dim=1))
        return x_mu

    def sample(self, z, y):
        for i, decoder in enumerate(self.decoder):
            z = decoder(z)
        return self.reconstruction(torch.cat([z, y], dim=1))