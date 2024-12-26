import torch
import math
import torch.nn as nn
import math

from architectures.common_utils import identity
from architectures.mlp import MLP, GaussianDist, CategoricalDistParams, TanhGaussianDistParams
import torch.nn.functional as F

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
            x = x.reshape(x.size(0), -1)
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
                 fc_output_activation=identity,
                 dropout_prob = 0
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
            output_activation=fc_output_activation,
            dropout_prob = dropout_prob
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









