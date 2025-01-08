import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from architectures.mlp import MLP
from utils.utils import anneal_coefficient
from architectures.m2_vae.stochastic import GaussianSample
from architectures.m2_vae.vae import VariationalAutoencoder
from architectures.m2_vae.distributions import log_standard_categorical
from architectures.m2_vae.vae import FeatureEncoder, Decoder, LadderEncoder, LadderDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, y_dim, hidden] = dims
        self.mlp = MLP(x_dim, y_dim, hidden, dropout_prob=0.20)
        # self.dense0 = nn.Linear(x_dim, 256)
        # self.dense1 = nn.Linear(256, 512)
        # self.logits = nn.Linear(512, y_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = F.softmax(x, dim=-1)
        return x


class DeepGenerativeModel(nn.Module):
    def __init__(self, dims):
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
        [self.x_dim, self.y_dim, self.h_dim, self.z_dim, self.classifier_hidden] = dims
        super(DeepGenerativeModel, self).__init__()
        self.encoder = FeatureEncoder([self.x_dim, self.h_dim])
        self.z_latent = GaussianSample(self.h_dim, self.z_dim)
        self.decoder = Decoder([self.z_dim, self.y_dim, self.h_dim, self.x_dim])
        self.classifier = Classifier([self.h_dim, self.y_dim, self.classifier_hidden])
        
        self.prior_c_logit = nn.Parameter(nn.Parameter(torch.full((self.y_dim,), 1 / self.y_dim)))


    def forward(self, x, y=None):
        # Add label and data and generate latent variable
        h = self.encoder(x)
        z, z_mu, z_log_var = self.z_latent(h[0])
        if y is None:
            y = self.classify(h[0])
        # Reconstruct data point from latent data and label
        x_reconstructed = self.decoder(z, y)

        return x_reconstructed, z, z_mu, z_log_var, y

    def classify(self, x):
        logits = self.classifier(x)
        return logits
    
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
        self.eval()
        with torch.no_grad():
            h = self.encoder(x_input)
            z, mu, logvar = self.z_latent(h[0])
            if use_mean_z:
                z = mu  # Use mean of q(z|x)
            else:
                z = self.reparameterize(mu, logvar)
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
    
    #Loss unction for labelled data
    def L(self,x, mu, logvar, z, y, kl_weight = 1):
        """
        Computes the loss function for GMVAE

        Args:
            x: Input images (B, 3, 40, 40)
            mu: Mean of q(z|x) (B, latent_dim)
            logvar: Log variance of q(z|x) (B, latent_dim)
            z: Sampled latent variables (B, latent_dim)
            y: Soft labels (B, num_classes), representing p(y|x)
        Returns:
            total_loss: The total loss for the batch
            loss_dict: Dictionary containing individual loss components
        """
        batch_size = x.size(0)
        device = x.device

        # Reconstruction loss (using soft labels c)
        # Compute x_recon directly using q_c_probs as the soft labels
        q_c_probs = self.classify(self.encoder(x)[0])
        x_recon = self.decoder(z, q_c_probs)  # Use soft labels in the decoder

        # Compute reconstruction loss per sample
        recon_loss_per_sample = F.binary_cross_entropy(x_recon, x, reduction='none')  # (B, C, H, W)
        recon_loss_per_sample = recon_loss_per_sample.view(batch_size, -1).sum(dim=1)  # Sum over pixels

        # Mean reconstruction loss over the batch
        recon_loss = recon_loss_per_sample.mean()

        # KL divergence for z (per sample, summed over latent dimensions)
        kl_z_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kl_z = kl_z_per_sample.mean()

        # KL divergence for c (per sample)
        # Assuming uniform prior p(c)
        log_q_c = torch.log(q_c_probs + 1e-8)  # (B, num_classes)
        self.prior_c = torch.softmax(self.prior_c_logit, dim=0)
        kl_c_per_sample = torch.sum(q_c_probs * (log_q_c - torch.log(self.prior_c)), dim=1)  # (B,)
        kl_c = kl_c_per_sample.mean()

        # Total VAE loss
        total_loss = recon_loss + kl_weight*kl_z + kl_weight*kl_c
        
        # y: Soft labels representing p(y|x) (B, num_classes)
        # Compute the expected log likelihood: E_{q(c|x)}[log p(y|c)]
        # Since p(y|c) = y_c (soft labels), and q(c|x) is available
        # We compute: -E_{q(c|x)}[log p(y|c)] = - q(c|x) * log(y)
        # Adding small value to prevent log(0)
        log_p_y_given_c = torch.log(y + 1e-8)  # (B, num_classes)
        # Compute the expected log likelihood term
        expected_log_p_y_given_c = torch.sum(q_c_probs * log_p_y_given_c, dim=1)  # (B,)
        # Classification loss is negative of the expected log likelihood
        cls_loss = -expected_log_p_y_given_c.mean()  # Sum over batch
        # Add classification loss to total loss
        total_loss += 1*cls_loss
        return total_loss, cls_loss

    # Loss function for unlabeled data
    def U(self,x, mu, logvar, z, q_c_probs, kl_weight = 1):
        """
        Computes the loss function for GMVAE

        Args:
            x: Input images (B, 3, 40, 40)
            mu: Mean of q(z|x) (B, latent_dim)
            logvar: Log variance of q(z|x) (B, latent_dim)
            z: Sampled latent variables (B, latent_dim)
            q_c_probs: Class probabilities q(c|x) (B, num_classes)
            y: Soft labels (B, num_classes), representing p(y|x)
        Returns:
            total_loss: The total loss for the batch
            loss_dict: Dictionary containing individual loss components
        """
        batch_size = x.size(0)
        device = x.device

        # Reconstruction loss (using soft labels c)
        # Compute x_recon directly using q_c_probs as the soft labels
        x_recon = self.decoder(z, q_c_probs)  # Use soft labels in the decoder

        # Compute reconstruction loss per sample
        recon_loss_per_sample = F.binary_cross_entropy(x_recon, x, reduction='none')  # (B, C, H, W)
        recon_loss_per_sample = recon_loss_per_sample.view(batch_size, -1).sum(dim=1)  # Sum over pixels

        # Total reconstruction loss summed over the batch
        recon_loss = recon_loss_per_sample.mean()

        # KL divergence for z (per sample, summed over latent dimensions)
        kl_z_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kl_z = kl_z_per_sample.mean()
        
        # KL divergence for c (per sample)
        # Assuming uniform prior p(c)
        log_q_c = torch.log(q_c_probs + 1e-8)  # (B, num_classes)
        self.prior_c = torch.softmax(self.prior_c_logit, dim=0)
        kl_c_per_sample = torch.sum(q_c_probs * (log_q_c - torch.log(self.prior_c)), dim=1)  # (B,)
        kl_c = kl_c_per_sample.mean()

        # Total VAE loss
        total_loss = recon_loss + kl_weight*kl_z + kl_weight*kl_c
        return total_loss, recon_loss, kl_z, kl_c
        
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