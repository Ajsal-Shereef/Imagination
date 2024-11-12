import wandb
import itertools
import math
import random
from time import time
from collections import OrderedDict
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

EPS = 1e-12


class Trainer:
    def __init__(self, model, optimizers, agent, device=None, recon_type=None, z_capacity=None,
                 u_capacity=None, c_gamma=None, entropy_gamma=None, bc_gamma=None, 
                 bc_threshold=None, save_freequency=100, model_save_path = 'models/parted_vae'):
        self.device = device

        self.model = model.to(self.device)
        self.agent = agent

        self.optimizer_warm_up, self.optimizer_imagination_net, self.optimizer_model = optimizers
        
        self.scheduler_warm_up = ReduceLROnPlateau(self.optimizer_warm_up, factor=0.5, patience=2, threshold=1e-1,
                                                   threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-06,
                                                   verbose=True)
        self.scheduler_imagination_net = ReduceLROnPlateau(self.optimizer_warm_up, factor=0.5, patience=2, threshold=1e-1,
                                                   threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-06,
                                                   verbose=True)
        self.scheduler_model = ReduceLROnPlateau(self.optimizer_model, factor=0.5, patience=2, threshold=1e-2,
                                                     threshold_mode='rel', cooldown=4, min_lr=0, eps=1e-07,
                                                     verbose=True)

        self.recon_type = recon_type
        self.save_freequency = save_freequency
        self.model_save_path = model_save_path

        self.z_capacity = z_capacity
        self.u_capacity = u_capacity
        self.c_gamma = c_gamma
        self.entropy_gamma = entropy_gamma
        self.bc_gamma = bc_gamma
        self.bc_threshold = bc_threshold

        # The following variable is used in computing KLD, it is computed here once, for speeding up
        self.u_kl_func_valid_indices = torch.zeros(self.model.sum_c_dims, dtype=torch.long, device=self.device, requires_grad=False)
        start = 0
        for value, disc_dim in enumerate(self.model.c_dims):
            self.u_kl_func_valid_indices[start:start + disc_dim] = value
            start += disc_dim

        # Used in computing KLs and Entropy for each random variable
        self.unwrap_mask = torch.zeros(self.model.sum_c_dims, self.model.sum_c_dims, self.model.c_count, device=self.device, requires_grad=False)
        start = 0
        for dim_idx, size in enumerate(self.model.c_dims):
            self.unwrap_mask[torch.arange(start, start + size), torch.arange(start, start + size), dim_idx] = 1
            start += size

        # Used in computing BC
        self.u_valid_prior_BC_mask = torch.zeros(self.model.sum_c_dims, self.model.sum_c_dims, device=self.device, requires_grad=False)
        start = 0
        for dim_idx, size in enumerate(self.model.c_dims):
            indices = itertools.product(range(start, start + size), range(start, start + size))
            self.u_valid_prior_BC_mask[list(zip(*indices))] = 1
            start += size
        self.u_valid_prior_BC_mask.tril_(diagonal=-1)

        self.num_steps = 0
        self.batch_size = None

    def train(self, data_loader, warm_up_loader=None, epochs=10, run_after_epoch=None, run_after_epoch_args=None):
        self.batch_size = data_loader.batch_size
        self.model.train()

        if warm_up_loader is None:
            print('No warm-up')

        epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
        for epoch in epoch_bar:
            warm_up_mean_loss, warm_up_imagined_loss, separated_mean_epoch_loss, imagined_losses = self._train_epoch(data_loader, warm_up_loader)
            epoch_bar.set_postfix({'Loss': separated_mean_epoch_loss[0], 
                              'Recon': separated_mean_epoch_loss[1], 
                              'Prior_loss': separated_mean_epoch_loss[2], 
                              'Class loss': separated_mean_epoch_loss[3],
                              'Prior class loss': separated_mean_epoch_loss[4]})
            wandb.log({
                'Classification loss' : warm_up_mean_loss,
                'Loss': separated_mean_epoch_loss[0], 
                'Recon': separated_mean_epoch_loss[1], 
                'Z KL divergence': separated_mean_epoch_loss[2], 
                'U KL divergence': separated_mean_epoch_loss[3],
                'Prior class loss': separated_mean_epoch_loss[4],
                'Class entropy loss' : separated_mean_epoch_loss[5],
                'Class interesection loss' : separated_mean_epoch_loss[6],
                'Imagined class loss' : imagined_losses[0],
                'Imagined state proximity loss' : imagined_losses[1],
                'Imagined state action loss' : imagined_losses[2],
            })
    
            if epoch%self.save_freequency==0:
                self.model.save(epoch)
            if warm_up_loader is not None:
                self.scheduler_warm_up.step(warm_up_mean_loss)
                self.scheduler_imagination_net.step(warm_up_imagined_loss)
            self.scheduler_model.step(separated_mean_epoch_loss[0])
            if run_after_epoch is not None:
                run_after_epoch(epoch, *run_after_epoch_args)
        self.save(epoch)

    def _train_epoch(self, data_loader, warm_up_loader=None):
        warm_up_loss, warm_up_imagined_loss, separated_sum_loss, imagined_losses = 0, 0, 0, 0
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, label) in enumerate(data_loader):
            warm_up_losses = self._warm_up(warm_up_loader)
            warm_up_loss += warm_up_losses[0]
            warm_up_imagined_loss += warm_up_losses[1]
            loss = self._train_iteration(data)
            separated_sum_loss += loss[0]
            imagined_losses += loss[1]

        return warm_up_loss / (batch_idx + 1), warm_up_imagined_loss/(batch_idx + 1), separated_sum_loss / len(data_loader.dataset), imagined_losses

    def _warm_up(self, loader):
        if not loader:
            return 0

        epoch_supervised_loss = 0
        epoch_supervised_imagined_loss = 0
        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)

            self.optimizer_warm_up.zero_grad()
            latent_dist = self.model.encode(data.float(), only_disc_dist=True)
            sum_ce = torch.sum(-1 * label * latent_dist['log_c'], dim=1)
            supervised_loss = torch.mean(sum_ce)
            supervised_loss.backward()
            # Clip gradients 
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            # Log the gradient norm to verify clipping
            # wandb.log({"gradient_norm": grad_norm})
            self.optimizer_warm_up.step()
            epoch_supervised_loss = epoch_supervised_loss + supervised_loss.item()
            
            self.optimizer_imagination_net.zero_grad()
            imagined_states = self.model.imagination_net(self.model.encoder(data.float()))
            imagined_states = imagined_states.view(-1, self.model.sum_c_dims, self.model.input_dim)
            # Get the indices where label_tensor is max
            max_indices = torch.argmax(label, dim=-1)  # [1000]
            # Expand indices to match the shape required for gather
            indices_expanded = max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 79)
            # Use gather to select values from data_tensor
            selected_imagined_states = torch.gather(imagined_states, 1, indices_expanded).squeeze(1)  # [1000, 79]
            imagined_latent_dist = self.model.encode(selected_imagined_states.float(), only_disc_dist=True)
            supervised_imagined_loss = torch.sum(-1 * label * imagined_latent_dist['log_c'], dim=1)
            supervised_imagined_loss = torch.mean(supervised_imagined_loss)
            supervised_imagined_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            self.optimizer_imagination_net.step()
            epoch_supervised_imagined_loss = epoch_supervised_imagined_loss + supervised_imagined_loss.item()
            
            return epoch_supervised_loss, epoch_supervised_imagined_loss

    def _train_iteration(self, data):
        self.num_steps += 1
        data = data.float().to(self.device)
        recon_batch, latent_dist = self.model(data.float())
        loss, separated_mean_loss = self._loss_function(data, recon_batch, latent_dist)
        if np.isnan(separated_mean_loss[0]):
            raise Exception('NaN!')

        self.optimizer_model.zero_grad()
        loss.backward()
        # Clip gradients 
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
        # Log the gradient norm to verify clipping
        # wandb.log({"gradient_norm": grad_norm})
        # Applying partially connected layers
        # with torch.no_grad():
        #     self.model.c_to_a_logit_pc.weight.grad.mul_(self.model.c_to_a_logit_mask)
        #     self.model.h_dot_a_to_u_mean_pc.weight.grad.mul_(self.model.h_dot_a_to_u_mask)
        #     self.model.h_dot_a_to_u_logvar_pc.weight.grad.mul_(self.model.h_dot_a_to_u_mask)
        self.optimizer_model.step()
        
        #Imagination loss calculation
        imagined_states = self.model.imagination_net(self.model.encoder(data))
        # imagined_states = imagined_states.view(-1, self.model.sum_c_dims, self.model.input_dim)
        self.model.eval()
        imagined_states_inference = self.model(imagined_states.view(-1, data.shape[-1]))[1]
        imagined_states_inference = imagined_states_inference['log_c']
        self.model.train()
        agent_action = self.agent.actor_local(data).to(self.device)
        imagined_state_action = self.agent.actor_local(imagined_states.view(-1, data.shape[-1])).to(self.device)
        # imagined_state_action.view(-1, self.model.sum_c_dims)
        
        loss, imagination_losses = self._imagination_loss(data, imagined_states, agent_action, imagined_state_action, imagined_states_inference)
        
        self.optimizer_imagination_net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
        self.optimizer_imagination_net.step()

        return separated_mean_loss * data.size(0), imagination_losses

    def _loss_function(self, data, recon_data, latent_dist):
        if self.recon_type.lower() == 'bce':
            recon_loss = F.binary_cross_entropy(recon_data, data, reduction='none')
        elif self.recon_type.lower() == 'mse':
            recon_loss = F.mse_loss(recon_data, data.float(), reduction='none')
        else:
            recon_loss = torch.abs(recon_data - data)
        recon_loss = torch.mean(torch.sum(recon_loss, dim=-1))

        tmp_zero = torch.zeros(1, device=self.device, requires_grad=False)
        z_kl, z_loss, each_c_kl, c_loss, each_c_entropy, c_entropy_loss, u_loss, each_u_expected_kl = 8 * [tmp_zero]
        bc, priors_intersection_loss = 2 * [tmp_zero]
        mean_bc = 0

        if self.model.has_indep:
            mean, logvar = latent_dist['z']
            z_each_dim_kl = self._kld_each_dim_with_standard_gaussian(mean, logvar)
            z_kl = torch.sum(z_each_dim_kl)
            cap_min, cap_max, num_iters, gamma = self.z_capacity
            cap_current = (cap_max - cap_min) * self.num_steps / float(num_iters) + cap_min
            cap_current = min(cap_current, cap_max)
            z_loss = gamma * torch.abs(cap_current - z_kl)
            # z_loss = gamma * z_kl
            

        if self.model.has_dep:
            each_c_kl = self._each_c_kl_loss(latent_dist['log_c'])
            c_loss = self.c_gamma * torch.sum(each_c_kl)

            each_c_entropy = self._each_c_entropy(latent_dist['log_c'])
            c_entropy_loss = self.entropy_gamma * torch.sum(each_c_entropy)

            each_u_expected_kl = self._each_u_expected_kl_loss(latent_dist['log_c'], latent_dist['u'])
            cap_min, cap_max, num_iters, gamma = self.u_capacity
            cap_current = (cap_max - cap_min) * self.num_steps / float(num_iters) + cap_min
            cap_current = min(cap_current, cap_max)
            u_loss = gamma * torch.abs(cap_current - torch.sum(each_u_expected_kl))
            # u_loss = gamma*torch.sum(each_u_expected_kl)

            bc = self._bhattacharyya_coefficient_inter_priors(self.model.u_prior_means, self.model.u_prior_logvars, self.u_valid_prior_BC_mask)
            priors_intersection_loss = self.bc_gamma * torch.sum(torch.clamp_min(bc - self.bc_threshold, min=0))
            mean_bc = (torch.sum(bc) / torch.sum(self.u_valid_prior_BC_mask)).item()
            
            
        # Total loss
        total_loss = recon_loss + z_loss + c_loss + u_loss + c_entropy_loss + priors_intersection_loss

        return (
            total_loss,
            np.array([total_loss.item(), recon_loss.item(), z_loss.item(), u_loss.item(), c_loss.item(),
                      c_entropy_loss.item(), priors_intersection_loss.item(),
                      z_kl.item(), torch.sum(each_u_expected_kl).item(), torch.sum(each_c_kl).item(),
                      torch.sum(each_c_entropy).item(), mean_bc,
                      *z_each_dim_kl.detach().cpu().numpy(),
                      *each_u_expected_kl.detach().cpu().numpy(),
                      *each_c_kl.detach().cpu().numpy(),
                      *each_c_entropy.detach().cpu().numpy()])
        )
        
    def _imagination_loss(self, data, imagined_states, agent_action, imagined_state_action, imagined_states_inference):
        states_repeated = data.repeat(self.model.c_dims[0], 1)
        # 1. Proximity loss between imagined state and original state
        prox_loss = F.mse_loss(imagined_states.view(-1, self.model.input_dim), states_repeated.float(), reduction='mean')
        
        # 2. Class consistency loss
        # Create the first 1000 rows as [1, 0]
        target1 = torch.tensor([[1, 0]]).repeat(imagined_states.shape[0], 1)  # Shape: [1000, 2]
        # Create the next 1000 rows as [0, 1]
        target2 = torch.tensor([[0, 1]]).repeat(imagined_states.shape[0], 1)  # Shape: [1000, 2]
        # Concatenate them along the first dimension
        target = torch.cat((target1, target2), dim=0)  # Shape: [2000, 2]
        cl_loss = F.cross_entropy(imagined_states_inference, target.float().to(self.device), reduction='mean')

        
        # 3. Action consistency loss
        actions_repeated = agent_action.repeat(self.model.c_dims[0], 1)
        action_loss = F.mse_loss(imagined_state_action, actions_repeated, reduction='mean')
        
        loss = cl_loss + action_loss + prox_loss
        return loss, np.array([cl_loss.item(), action_loss.item(), prox_loss.item()])

    def _kld_each_dim_with_standard_gaussian(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        return kl_means

    def _bhattacharyya_coefficient_inter_priors(self, mu, logvar, mask):
        variance = torch.exp(logvar)
        avg_var = 0.5 * (variance.unsqueeze(0) + variance.unsqueeze(1))  # n_priors, n_priors, d
        inv_avg_var = 1 / (avg_var + EPS)
        diff_mean = mu.unsqueeze(0) - mu.unsqueeze(1)
        db_first_term = 1/8 * torch.sum(diff_mean * inv_avg_var * diff_mean, dim=2)  # n_priors, n_priors
        db_second_term = 0.5 * (torch.sum(torch.log(avg_var + EPS), dim=2)
                                - 0.5 * (torch.sum(logvar, dim=1).unsqueeze(0) + torch.sum(logvar, dim=1).unsqueeze(1)))
        db = db_first_term + db_second_term
        bc = torch.exp(-db)
        valid_bc = bc.mul(mask)
        return valid_bc

    def _kld_each_dim_data_and_priors(self, prior_means, prior_logvars, batch_mean, batch_logvar, parts_count, valid_indices):
        n_priors, d = prior_means.size()

        batch_logvar = batch_logvar.view(-1, 1, parts_count, d).expand(-1, n_priors, -1, -1)[:, torch.arange(n_priors), valid_indices, :]
        batch_var = torch.exp(batch_logvar)  # batch_size, n_priors, d

        diff_mean_with_invalid_items = prior_means.unsqueeze(0).unsqueeze(2) - batch_mean.view(-1, 1, parts_count, d)  # batch_size, n_priors, disc_count, d
        diff_mean = diff_mean_with_invalid_items[:, torch.arange(n_priors), valid_indices, :]  # batch_size, n_priors, d

        priors_unsqueezed_inv_var = torch.exp(-1 * prior_logvars).unsqueeze(0)  # 1, n_priors, d

        return 0.5 * (
                prior_logvars.unsqueeze(0)  # 1, n_priors, d
                - batch_logvar  # batch_size, n_priors, d
                - 1
                + priors_unsqueezed_inv_var * batch_var  # batch_size, n_priors, d
                + diff_mean * priors_unsqueezed_inv_var * diff_mean  # batch_size, n_priors, d
        )  # batch_size, n_priors, d

    def _each_u_expected_kl_loss(self, log_q_cs_given_x, u_dist):
        each_dim_kld = self._kld_each_dim_data_and_priors(self.model.u_prior_means, self.model.u_prior_logvars,
                                                          u_dist[0], u_dist[1],
                                                          self.model.c_count, self.u_kl_func_valid_indices)
        kld = torch.sum(each_dim_kld, dim=2)
        kld_dot_prob = torch.exp(log_q_cs_given_x) * kld
        unwrapped_kld_dot_prob = kld_dot_prob.unsqueeze(1).unsqueeze(2).matmul(self.unwrap_mask.unsqueeze(0)).squeeze(2)
        expected_kld = torch.sum(unwrapped_kld_dot_prob, dim=1)  # batch_size, disc_count
        each_u_expected_kl_loss = torch.mean(expected_kld, dim=0)
        return each_u_expected_kl_loss

    def _each_c_kl_loss(self, log_q_cs_given_x):  # log_q_cs_given_x's size is batch * sum_disc_dim
        log_q_cs = torch.logsumexp(log_q_cs_given_x, dim=0) - math.log(log_q_cs_given_x.size(0))
        q_log_q_on_p = torch.exp(log_q_cs) * (log_q_cs - torch.log(self.model.c_priors + EPS))
        unwrapped_q_log_q_on_p = q_log_q_on_p.matmul(self.unwrap_mask)
        each_c_kl_loss = torch.sum(unwrapped_q_log_q_on_p, dim=0)
        return each_c_kl_loss

    def _each_c_entropy(self, log_q_cs_given_x):
        q_log_q = torch.exp(log_q_cs_given_x) * log_q_cs_given_x
        unwrapped_q_log_q = q_log_q.matmul(self.unwrap_mask)
        batch_each_c_neg_entropy = torch.sum(unwrapped_q_log_q, dim=0)
        each_c_entropy = -1 * torch.mean(batch_each_c_neg_entropy, dim=0)
        return each_c_entropy
    
