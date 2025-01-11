import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable
from architectures.mlp import Linear
from architectures.film import FiLMLayer
from architectures.cnn import CNNLayer, CNN
from utils.utils import custom_soft_action_encoding
from architectures.m2_vae.distributions import log_gaussian, log_standard_gaussian
from architectures.m2_vae.stochastic import GaussianSample, GaussianMerge, GumbelSoftmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)

        return x

class FeatureEncoder(nn.Module):
    def __init__(self, dims):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(FeatureEncoder, self).__init__()

        [input_dim, h_dim, feature_encoder_channel_dim] = dims
        conv1 = CNNLayer(input_dim, feature_encoder_channel_dim[0], 3, 2)
        conv2 = CNNLayer(feature_encoder_channel_dim[0], feature_encoder_channel_dim[1], 3, 2)
        conv3 = CNNLayer(feature_encoder_channel_dim[1], feature_encoder_channel_dim[2], 3)
        conv4 = CNNLayer(feature_encoder_channel_dim[2], feature_encoder_channel_dim[3], 3)
        conv_feature = Linear(1600, h_dim, dropout_prob=0.20)
        self.feature_encoder = CNN([conv1, conv2, conv3, conv4], conv_feature)
        

    def forward(self, x):
        return self.feature_encoder(x)
    
# class FiLMLayer(nn.Module):
#     def __init__(self, input_dim, feature_dim):
#         super(FiLMLayer, self).__init__()
#         self.gamma = nn.Linear(input_dim, feature_dim)
#         self.beta = nn.Linear(input_dim, feature_dim)

#     def forward(self, x, y):
#         gamma = self.gamma(y)
#         beta = self.beta(y)
#         return gamma.view(-1, 1, 1, 1) * x + beta.view(-1, 1, 1, 1)

class Decoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network
        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, y_dim, h_dim, x_dim] = dims
        self.z_dim = z_dim
        self.num_goals = y_dim
        self.fc_cnn1 = Linear(z_dim, h_dim, dropout_prob=0.20)
        self.fc_cnn2 = Linear(h_dim, 1600, dropout_prob=0.20)

        # FiLM layers for conditioning
        self.film1 = FiLMLayer(1600, y_dim)
        self.film2 = FiLMLayer(32*9*9, y_dim)
        self.film3 = FiLMLayer(16*19*19, y_dim)

        # Transposed convolutional layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)  # 64x5x5 -> 32x9x9
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0)  # 32x16x16 -> 16x19x19
        self.deconv3 = nn.ConvTranspose2d(16, x_dim, kernel_size=4, stride=2, padding=0)  # 16x40x40 -> x_dimx40x40
        
        # Activation function
        self.intermediate_activation = nn.LeakyReLU()
        self.last_activation = nn.Sigmoid()


    def forward(self, z, y):
        # Create soft action encoding for labels
        # y = torch.tensor(custom_soft_action_encoding(y, self.num_goals, self.z_dim)).to(z.device).float()

        # Initial linear layers
        linear_feature = self.fc_cnn1(z)
        linear_feature = self.fc_cnn2(linear_feature)

        # Condition each layer with FiLM
        x = self.film1(linear_feature, y)
        x = self.intermediate_activation(self.deconv1(x.view(-1, 64, 5, 5)))  # ConvTranspose2d(64 -> 32)
        x = self.film2(x.view(-1, 32*9*9), y)
        x = self.intermediate_activation(self.deconv2(x.view(-1, 32, 9, 9)))  # ConvTranspose2d(32 -> 16)
        x = self.film3(x.view(-1, 16*19*19), y)
        x = self.last_activation(self.deconv3(x.view(-1, 16, 19, 19)))  # ConvTranspose2d(16 -> x_dim)
        return x


# class Decoder(nn.Module):
#     def __init__(self, dims):
#         """
#         Generative network

#         Generates samples from the original distribution
#         p(x) by transforming a latent representation, e.g.
#         by finding p_θ(x|z,y).

#         :param dims: dimensions of the networks
#             given by the number of neurons on the form
#             [latent_dim, [hidden_dims], input_dim].
#         """
#         super(Decoder, self).__init__()

#         [z_dim, y_dim, x_dim] = dims
#         self.z_dim = z_dim
#         self.num_goals = y_dim
#         # self.film = FiLMLayer(z_dim, z_dim)
        
#         self.fc_cnn1 = Linear(z_dim*2, 256)
#         self.fc_cnn2 = Linear(256, 1600)
#         self.net = nn.Sequential(
#             # First layer: Upsample to 16x16
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # 64x3x3 -> 32x16x16
#             nn.LeakyReLU(),
            
#             # Second layer: Upsample to 40x40
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),  # 32x16x16 -> 3x40x40
#             nn.LeakyReLU(),  # Optional: Apply activation for output normalization
            
#             # Third layer: Upsample to 40x40
#             nn.ConvTranspose2d(16, x_dim, kernel_size=4, stride=2, padding=0),  # 32x16x16 -> 3x40x40
#             nn.LeakyReLU()  # Optional: Apply activation for output normalization
#         )

#     def forward(self, z, y):
#         action_encoded = torch.tensor(custom_soft_action_encoding(y, self.num_goals, self.z_dim)).to(device).float()
#         x = torch.cat((z, action_encoded), 1)
#         # x = self.film(z,action_encoded)
#         linear_feature = self.fc_cnn1(x)
#         linear_feature = self.fc_cnn2(linear_feature)
#         linear_feature = linear_feature.view(linear_feature.shape[0], 64, 5, 5)
#         return self.net(linear_feature)


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()

        [x_dim, y_dim, z_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, z_dim])
        self.decoder = Decoder([z_dim, y_dim, x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = pz - qz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        super(GumbelAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Perceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), x_dim], output_activation=F.sigmoid)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        k = Variable(torch.FloatTensor([self.z_dim]), requires_grad=False)
        kl = qz * (torch.log(qz + 1e-8) - torch.log(1.0/k))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1):
        x = self.encoder(x)

        sample, qz = self.sampler(x, tau)
        self.kl_divergence = self._kld(qz)

        x_mu = self.decoder(sample)

        return x_mu

    def sample(self, z):
        return self.decoder(z)


class LadderEncoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
        return x, self.sample(x)


class LadderDecoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()

        [self.z_dim, h_dim, x_dim] = dims

        self.linear1 = nn.Linear(x_dim, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, self.z_dim)

        self.linear2 = nn.Linear(x_dim, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            # Sample from this encoder layer and merge
            z = self.linear1(x)
            z = F.leaky_relu(self.batchnorm1(z), 0.1)
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.leaky_relu(self.batchnorm2(z), 0.1)
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self, dims):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        [x_dim, z_dim, h_dim] = dims
        super(LadderVariationalAutoencoder, self).__init__([x_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0], h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for encoder in self.encoder:
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

        x_mu = self.reconstruction(z)
        return x_mu

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)