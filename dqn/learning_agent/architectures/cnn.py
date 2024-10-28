import torch
import math
import torch.nn as nn
import math

from learning_agent.common_utils import identity
from learning_agent.architectures.mlp import MLP, GaussianDist, CategoricalDistParams, TanhGaussianDistParams
from utils.utils import get_device, custom_action_encoding
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_layer = nn.Conv2d,
        pre_activation_fn=identity,
        activation_fn=nn.LeakyReLU(),
        post_activation_fn=identity,
        gain = math.sqrt(2)
    ):
        super(CNNLayer, self).__init__()
        self.cnn = conv_layer(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.orthogonal_(self.cnn.weight, gain)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        x = self.batch_norm(x)
        return x


class CNN(nn.Module):
    """ Baseline of Convolution neural network. """
    def __init__(self, cnn_layers, fc_layers):
        """
        cnn_layers: List[CNNLayer]
        fc_layers: MLP
        """
        super(CNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x, is_flatten=True):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        # flatten x
        if is_flatten:
            x = x.view(x.size(0), -1)
        return x
    
    def get_cnn_feature(self, input, cnn_layer, is_flatten=True):
        if len(input.size()) == 3:
            input = input.unsqueeze(0)
            output = cnn_layer(input)
        if len(input.size()) == 5:
            b,t,c,h,w = input.size()
            input = input.view(b*t, c,h,w)
            output = cnn_layer(input)
            output = output.view(b,t,output.size()[1],output.size()[2],output.size()[3])
        return output

    def forward(self, x, is_flatten = True, **fc_kwargs):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x, is_flatten)
        if is_flatten:
            if self.fc_layers:
                fc_out = self.fc_layers(x, **fc_kwargs)
            return fc_out, x
        else:
            return None, x
    
class Conv2d_MLP_Model(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_MLP_Model, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]
        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = MLP(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, is_flatten = True):
        return self.conv_mlp.forward(x, is_flatten)


class MLP_Model(nn.Module):
    """ Fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self, hidden_units):
        super(MLP_Model, self).__init__()
        self.fc_layers = MLP(
            input_size=25,
            output_size=4,
            hidden_sizes=hidden_units,
            hidden_activation=torch.relu,
            output_activation=identity
        )

    def forward(self, x):
        return self.fc_layers(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)
            

# class VAE(nn.Module):
#     def __init__(self, 
#                  input_dim, 
#                  latent_dim,
#                  encoder_channels,
#                  encoder_kernel_sizes,
#                  encoder_strides,
#                  encoder_paddings,
#                  encoder_img_dim,
#                  decoder_channels,
#                  decoder_kernel_sizes,
#                  decoder_strides,
#                  decoder_paddings):
#         super(VAE, self).__init__()
        
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.encoder_channels = encoder_channels
#         self.encoder_img_dim = encoder_img_dim
        
#         in_channels = [input_dim] + encoder_channels[:-1]
        
#         #Encoder activation
#         activation_fns = [torch.relu for _ in range(len(encoder_strides))]
#         post_activation_fns = [identity for _ in range(len(encoder_strides))]
        
#         #Encoder conv layers
#         encoder_conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
#                                 kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
#                        for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, encoder_channels, encoder_kernel_sizes, 
#                                                                 encoder_strides, encoder_paddings, activation_fns, post_activation_fns)]
        
#         # Encoder layers
#         self.encoder = nn.Sequential()
#         for i, cnn_layer in enumerate(encoder_conv_layers):
#             self.encoder.add_module("cnn_{}".format(i), cnn_layer)
        
#         self.fc_mean = nn.Linear(encoder_channels[-1] * encoder_img_dim * encoder_img_dim, latent_dim)
#         self.fc_logvar = nn.Linear(encoder_channels[-1] * encoder_img_dim * encoder_img_dim, latent_dim)
        
#         # Decoder layers
#         conv = [nn.ConvTranspose2d for _ in range(len(decoder_strides))]
#         activation_fns[-1] = torch.sigmoid
#         out_channels = decoder_channels[1:] + [input_dim]
#         self.decoder_fc = nn.Linear(latent_dim, encoder_channels[-1] * encoder_img_dim * encoder_img_dim)
        
#         #Decoder conv layers
#         decoder_conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
#                                 kernel_size=k, stride=s, padding=p, conv_layer=c, activation_fn=a_fn, post_activation_fn=p_fn)
#                        for (ic, oc, k, s, p, c, a_fn, p_fn) in zip(decoder_channels, out_channels, decoder_kernel_sizes,
#                                                                    decoder_strides, decoder_paddings, conv, activation_fns, post_activation_fns)]
        
#         # Encoder layers
#         self.decoder = nn.Sequential()
#         for i, cnn_layer in enumerate(decoder_conv_layers):
#             self.decoder.add_module("cnn_{}".format(i), cnn_layer )
    
#     def encode(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         mean = self.fc_mean(x)
#         logvar = self.fc_logvar(x)
#         return mean, logvar
    
#     def reparameterize(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mean + eps * std
#         return z
    
#     def decode(self, z):
#         z = self.decoder_fc(z)
#         z = z.view(z.size(0), self.encoder_channels[-1], self.encoder_img_dim, self.encoder_img_dim)  # Reshape the tensor
#         x_hat = self.decoder(z)
#         return x_hat
    
#     def forward(self, x):
#         #TODO Add a flattened layer and try adding convolution layer
#         mean, logvar = self.encode(x)
#         z = self.reparameterize(mean, logvar)
#         x_hat = self.decode(z)
#         return x_hat, z, mean, logvar

class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # Decoder layers
        self.fc3 = nn.Linear(latent_size, 512)
        self.fc4 = nn.Linear(512, 256 * 5 * 5)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=0)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1, 256, 5, 5)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        z = torch.sigmoid(self.deconv5(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar


# class VAE(nn.Module):
#     def __init__(self, latent_dim):
#         super(VAE, self).__init__()

#         self.encoder = models.resnet18(pretrained=False)
#         self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.encoder.fc = nn.Linear(512, latent_dim * 2)

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 40 * 40),
#             nn.Sigmoid()
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         epsilon = torch.randn_like(std)
#         z = mu + epsilon * std
#         return z

#     def forward(self, x):
#         x = self.encoder(x)
#         mu, logvar = torch.chunk(x, 2, dim=1)
#         z = self.reparameterize(mu, logvar)
#         x = self.decoder(z)
#         x = x.view(-1, 1, 40, 40)
#         return x, z, mu, logvar

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def forward(self, input, size=1024):
#         return input.view(input.size(0), size, 1, 1)

# class VAE(nn.Module):
#     def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),
#             nn.ReLU(),
#             Flatten()
#         )
        
#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)
        
#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
#             nn.Sigmoid(),
#         )
        
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         # return torch.normal(mu, std)
#         esp = torch.randn(*mu.size())
#         z = mu + std * esp
#         return z
    
#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar

#     def encode(self, x):
#         h = self.encoder(x)
#         z, mu, logvar = self.bottleneck(h)
#         return z, mu, logvar

#     def decode(self, z):
#         z = self.fc3(z)
#         z = self.decoder(z)
#         return z

#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         z = self.decode(z)
#         return z, mu, logvar

    
##########################VQ-VAE Starts Here#######################################
"""Code adapted from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=2qnYPliTbizh"""

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    
class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, input_dim, decay=0):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(input_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                out_channels = input_dim)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon,quantized, loss, perplexity

##########################VQ-VAE Ends Here#########################################        

    
        
    
class Conv2d_MLP_Gaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Gaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = GaussianDist(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self.conv_mlp.forward(x)


class Conv2d_MLP_Categorical(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Categorical, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = CategoricalDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_categorical_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, deterministic=False):
        return self.conv_categorical_mlp.forward(x, deterministic=deterministic)


class Conv2d_MLP_TanhGaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_TanhGaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = TanhGaussianDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_tanh_gaussian_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        return self.conv_tanh_gaussian_mlp.forward(x, epsilon=1e-6, deterministic=False, reparameterize=True)


class Conv2d_Flatten_MLP(Conv2d_MLP_Model):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP, self).__init__(input_channels=input_channels,
                                                 fc_input_size=fc_input_size,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)

    def forward(self, *args):
        obs_x, augment_features = args
        cnn_features = self.conv_mlp.get_cnn_features(obs_x)
        features = torch.cat((cnn_features, augment_features), dim=1)
        return self.conv_mlp.fc_layers(features)









