
"""
Code adapted and modified from https://github.com/sinahmr/parted-vae
"""

import torch
import itertools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn
from architectures.mlp import MLP, Linear
EPS = 1e-12


def my_tanh(x):
    return 2 * torch.tanh(x) - 1  # std = e^{0.5 * logvar}: [0.22, 1.64]


class PartialLogSoftmax:
    def __init__(self, dimensions, device):
        self.device = device
        self.sum_dims = sum(dimensions)
        self.minus_inf = torch.tensor([float('-inf')], device=self.device, requires_grad=False)
        self.eps = torch.tensor([EPS], device=self.device, requires_grad=False)

        self.unwrap_logits_mask = torch.zeros(1, self.sum_dims, self.sum_dims, len(dimensions), device=self.device, requires_grad=False)
        self.scatter_log_denominator_mask = torch.zeros(len(dimensions), self.sum_dims, device=self.device, requires_grad=False)

        start = 0
        for dim_idx, size in enumerate(dimensions):
            self.unwrap_logits_mask[0, torch.arange(start, start + size), torch.arange(start, start + size), dim_idx] = 1
            self.scatter_log_denominator_mask[dim_idx, torch.arange(start, start + size)] = 1
            start += size

    def __call__(self, logits):
        logits = torch.where(logits == 0, self.eps, logits)  # Later on, we replace 0s with -inf, but 0s in logits have meaning and shouldn't be replaced
        unwrapped_logits = logits.unsqueeze(1).unsqueeze(2).matmul(self.unwrap_logits_mask).squeeze(2)  # batch_size, sum_dims, count_dims
        unwrapped_logits = torch.where(unwrapped_logits == 0, self.minus_inf, unwrapped_logits)
        log_denominator = torch.logsumexp(unwrapped_logits, dim=1)
        log_denominator = log_denominator.matmul(self.scatter_log_denominator_mask)
        log_softmax = logits - log_denominator
        return log_softmax


class VAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, output_size, latent_spec, c_priors, save_dir, imagination_net_config, 
                 env, device, temperature=0.67):
        super(VAE, self).__init__()
        self.device = device
        self.save_dir = save_dir
        
        self.env = env
        
        # Parameters
        self.has_indep = latent_spec.get('z', 0) > 0
        self.has_dep = len(latent_spec.get('c', [])) > 0
        if self.has_dep and latent_spec.get('single_u', 0) < 0:
            raise RuntimeError('Model has c variables but u_dim is 0')

        self.latent_spec = latent_spec
        self.input_dim = input_dim
        self.temperature = temperature
        self.hidden_dim = encoder_hidden_dim
        
        # Scale factor to avoid overly large attention scores
        self.scale = 1.0 / (output_size ** 0.5)

        self.z_dim, self.single_dep_cont_dim, self.u_dim = 0, 0, 0
        self.c_dims, self.c_count, self.sum_c_dims = 0, 0, 0
        self.c_priors = []
        if self.has_indep:
            self.z_dim = self.latent_spec['z']

        if self.has_dep:
            self.c_priors = torch.tensor(sum(c_priors, []), device=self.device, requires_grad=False)
            self.c_dims = self.latent_spec['c']
            self.c_count = len(self.c_dims)
            self.sum_c_dims = sum(self.c_dims)
            self.single_u_dim = self.latent_spec['single_u']
            self.u_dim = self.single_u_dim * self.c_count

        self.latent_dim = self.z_dim + self.u_dim

        # Encoder
        self.encoder = MLP(input_size = self.input_dim,
                           output_size = output_size,
                           hidden_sizes = self.hidden_dim,
                           output_activation = F.relu,
                           dropout_prob = 0.0)
        
        self.imagination_net = MLP(input_size = output_size,
                                   output_size = self.sum_c_dims*self.input_dim,
                                   hidden_sizes = self.hidden_dim,
                                   output_activation = F.relu,
                                   dropout_prob = 0.0)
        
        # Latent Space
        # FC: Fully Connected, PC: Partially Connected
        if self.has_indep:
            self.z_mean_fc = nn.Linear(output_size, self.z_dim)
            self.z_logvar_fc = nn.Linear(output_size, self.z_dim)

        if self.has_dep:
            self.h_to_c_logit_fc = nn.Linear(output_size, self.sum_c_dims) #Softmax is performed in the forward method

            self.c_to_a_logit_pc = nn.Linear(self.sum_c_dims, self.c_count * output_size)  # A Sigmoid should be placed after this layer
            # self.c_to_a_logit_mask = torch.zeros(self.c_count * output_size, self.sum_c_dims, requires_grad=False)
            # h_start = 0
            # for i, h_size in enumerate(self.c_dims):
            #     v_start, v_end = i * output_size, (i + 1) * output_size
            #     h_end = h_start + h_size
            #     indices = itertools.product(range(v_start, v_end), range(h_start, h_end))
            #     self.c_to_a_logit_mask[list(zip(*indices))] = 1  # It actually unzips :D
            #     h_start = h_end
            # with torch.no_grad():
            #     self.c_to_a_logit_pc.weight.mul_(self.c_to_a_logit_mask)

            self.h_dot_a_to_u_mean_pc = nn.Linear(self.c_count * output_size, self.u_dim)
            self.h_dot_a_to_u_logvar_pc = nn.Linear(self.c_count * output_size, self.u_dim)
            # self.h_dot_a_to_u_mask = torch.zeros(self.u_dim, self.c_count * output_size, requires_grad=False)
            # for i, dim in enumerate(self.c_dims):
            #     v_start, v_end = i * self.single_u_dim, (i + 1) * self.single_u_dim
            #     h_start, h_end = i * output_size, (i + 1) * output_size
            #     indices = itertools.product(range(v_start, v_end), range(h_start, h_end))
            #     self.h_dot_a_to_u_mask[list(zip(*indices))] = 1
            # with torch.no_grad():
            #     self.h_dot_a_to_u_mean_pc.weight.mul_(self.h_dot_a_to_u_mask)
            #     self.h_dot_a_to_u_logvar_pc.weight.mul_(self.h_dot_a_to_u_mask)

            # # These lines should be after the multiplications in torch.no_grad(), because model (and therefore h_dot_e_to_u_mean_pc.weight) hasn't gone to GPU yet
            # self.c_to_a_logit_mask = self.c_to_a_logit_mask.to(self.device)
            # self.h_dot_a_to_u_mask = self.h_dot_a_to_u_mask.to(self.device)

        # Decoder
        self.decoder = MLP(input_size = self.latent_dim,
                           output_size = input_dim,
                           hidden_sizes = decoder_hidden_dim,
                           output_activation = F.relu,
                           dropout_prob = 0.0)

        # Define psi network
        self.u_prior_means = nn.Parameter(torch.randn(self.sum_c_dims, self.single_u_dim), requires_grad=True)
        self.u_prior_logvars_before_tanh = nn.Parameter(torch.randn(self.sum_c_dims, self.single_u_dim), requires_grad=True)

        self.sigmoid_coef = 1

        self.logsoftmaxer = PartialLogSoftmax(self.c_dims, device=self.device)

        self.to(self.device)

    @property
    def u_prior_logvars(self):
        return my_tanh(self.u_prior_logvars_before_tanh)

    def my_sigmoid(self, x):
        if not self.training or self.sigmoid_coef > 8:
            return torch.sigmoid(8 * x)
        if self.sigmoid_coef < 8:
            self.sigmoid_coef += 2e-4
        return torch.sigmoid(self.sigmoid_coef * x)

    def encode(self, x, only_disc_dist=False):  # x: (N, C, H, W)
        batch_size = x.size()[0]

        h = self.encoder(x)
        latent_dist = dict()
        if self.has_dep:
            c_logit = self.h_to_c_logit_fc(h)
            latent_dist['log_c'] = self.logsoftmaxer(c_logit)
            if only_disc_dist:
                return latent_dist

            sampled_c = self.sample_gumbel_partial_softmax(c_logit)  # One hot (sort of)
            a_logit = self.c_to_a_logit_pc(sampled_c)
            
            # Compute the attention scores between h and a
            scores = torch.matmul(h, a_logit.transpose(-2, -1)) * self.scale
            attention_weights = F.softmax(scores, dim=-1)
            # Apply attention weights to h
            attended_h = torch.matmul(attention_weights, h)
            
            latent_dist['u'] = [self.h_dot_a_to_u_mean_pc(attended_h), self.h_dot_a_to_u_logvar_pc(attended_h)]

        if self.has_indep:
            latent_dist['z'] = [self.z_mean_fc(h), self.z_logvar_fc(h)]

        return latent_dist

    def reparameterize(self, latent_dist):
        latent_sample = list()
        for mean, logvar in [latent_dist['z'], latent_dist['u']]:
            sample = self.sample_normal(mean, logvar)
            latent_sample.append(sample)
        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size(), device=self.device).normal_()
            return mean + std * eps
        else:
            return mean

    def sample_gumbel_partial_softmax(self, c_logit, soft=1):
        if soft:
            unif = torch.rand(c_logit.size(), device=self.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            logit = (c_logit + gumbel) / self.temperature
            return torch.exp(self.logsoftmaxer(logit))
        else:
            alphas = torch.exp(self.logsoftmaxer(c_logit))
            one_hot_samples = torch.zeros(alphas.size(), device=self.device)
            start = 0
            for size in self.c_dims:  # Here speed is not that important
                alpha = alphas[:, start:start+size]
                _, max_alpha = torch.max(alpha, dim=1)
                one_hot_sample = torch.zeros(alpha.size(), device=self.device)
                one_hot_sample.scatter_(1, max_alpha.view(-1, 1).data, 1)
                one_hot_samples[:, start:start+size] = one_hot_sample
                start += size
            return one_hot_samples

    def decode(self, latent_sample):
        return self.decoder(latent_sample)

    def forward(self, x):
        latent_dist = self.encode(x)
        latent_sample = self.reparameterize(latent_dist)
        return self.decode(latent_sample), latent_dist
    
    def load(self, path_to_model):
        old_state_dict = torch.load(path_to_model, map_location=self.device)
        # state_dict = OrderedDict()
        # for k, v in old_state_dict.items():
        #     k = k.replace('c1_dz_prior', 'u_prior')
        #     k = k.replace('mean_fc', 'z_mean_fc')
        #     k = k.replace('logvar_fc', 'z_logvar_fc')
        #     k = k.replace('c_to_e_logit_pc', 'c_to_a_logit_pc')
        #     k = k.replace('h_dot_e_to_dz_mean_pc', 'h_dot_a_to_u_mean_pc')
        #     k = k.replace('h_dot_e_to_dz_logvar_pc.0', 'h_dot_a_to_u_logvar_pc')
        #     if 'c2_dz_prior' in k:
        #         continue
        #     state_dict[k] = v

        self.load_state_dict(old_state_dict)
        self.sigmoid_coef = 8.
        print(f"[INFO] Model loaded from {path_to_model}")
        return self
    
    def save(self, epochs):
        torch.save(self.state_dict(), f'{self.save_dir}/parted_vae_{epochs}.pth')
        print(f"[INFO] model save to {self.save_dir}/parted_vae_{epochs}.pth")
