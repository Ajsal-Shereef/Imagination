import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils.utils import *
import torch.optim as optim
import torch.distributed as dist
from architectures.mlp import MLP
from torch.nn.utils import clip_grad_norm_
from dqn.learning_agent.agent import Agent
from dqn.learning_agent import common_utils
from dqn.dqn_utils import calculate_dqn_loss
from dqn.learning_agent.replay_buffer import ReplayBuffer
from dqn.learning_agent.priortized_replay_buffer import PrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(nn.Module):
    """
    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy controller_policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, save_dir):
        """Initialize."""
        super(DQNAgent, self).__init__()
        
        self.env = env
        self.args = args

        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0
        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.per_beta = hyper_params.per_beta
        self.use_n_step = hyper_params.n_step > 1
        self.use_prioritized = hyper_params.use_prioritized

        self.max_epsilon = hyper_params.max_epsilon
        self.min_epsilon = hyper_params.min_epsilon
        self.epsilon = hyper_params.max_epsilon

        self._initialize()
        self._init_network()
        
        self.save_dir = save_dir

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # replay memory for a single step
        if self.use_prioritized:
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                alpha=self.hyper_params.per_alpha,
            )
        # use ordinary replay buffer
        else:
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=self.hyper_params.batch_size,
                gamma=self.hyper_params.gamma,
            )

        # replay memory for multi-steps
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=self.hyper_params.batch_size,
                n_step=self.hyper_params.n_step,
                gamma=self.hyper_params.gamma,
            )
            
    
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = MLP(input_size=self.network_cfg.fc_input_size,
                       output_size=self.env.action_space.n,
                       hidden_sizes = self.network_cfg.hidden_size).to(device)
        self.dqn_target = MLP(input_size=self.network_cfg.fc_input_size,
                              output_size=self.env.action_space.n,
                              hidden_sizes = self.network_cfg.hidden_size).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        for param in self.dqn_target.parameters():
            param.requires_grad = False
        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # init network from file
        #self._init_from_file()

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)
            
    def reset_frame_array(self):
        self.frame_array = []

    def get_action(self, state, epsilon=0):
        """Select an action from the input space."""
        if (epsilon > np.random.random() or self.total_step < self.hyper_params.init_random_actions):
            selected_action = self.env.action_space.sample()
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state).unsqueeze(0).float().to(device)
            with torch.no_grad():
                policy_dqn_output = self.dqn(state)
            policy_dqn_output = policy_dqn_output.squeeze(0)
            if policy_dqn_output.dim() != 1:
                return torch.argmax(policy_dqn_output, dim=-1)
            policy_dqn_output = policy_dqn_output.detach().cpu().numpy()
            selected_action = np.argmax(policy_dqn_output)
        return selected_action
    
    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info

    def add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_dqn_loss(self, experiences, gamma):
        """Return element-wise dqn loss and Q-values."""
        return calculate_dqn_loss(
            model=self.dqn,
            target_model=self.dqn_target,
            experiences=experiences,
            gamma=gamma,
            use_double_q_update=self.hyper_params.use_double_q_update,
            reward_clip=self.hyper_params.reward_clip,
            reward_scale=self.hyper_params.reward_scale,
        )

    def update_model(self):
        """Train the model after each episode."""
        # 1 step loss
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices = experiences_one_step[-3:-1]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights/torch.sum(weights)*float(self.hyper_params.batch_size)
        else:
            indices = np.random.choice(len(self.memory), size=self.hyper_params.batch_size, replace=False)
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)

        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_one_step, self.hyper_params.gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(experiences_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            # mix of 1-step and n-step returns
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # total loss
        loss = dq_loss

        # q_value regularization (not used when w_q_reg is set to 0)
        if self.optim_cfg.w_q_reg > 0:
            q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg
            loss = loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        #common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)
        if self.total_step % self.hyper_params.target_network_update_interval == 0:
            common_utils.hard_update(self.dqn, self.dqn_target)
        
        # update priorities in PER
        if self.use_prioritized:
            loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            if (new_priorities <= 0).any().item():
                print('[ERROR] new priorities less than 0. Loss info: ', str(loss_for_prior))

            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.iteration_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return loss.item(), q_values.mean().item()
    
    def load_params(self, path):
        """Load model and optimizer parameters."""
        #path = path + '/dqn_model' + '.tar'
        params = torch.load(path, map_location=device)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the DQN model and optimizer from", path)
    
    def save_models(self, checkpoint, saved_dir, prefix):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """

        path = saved_dir + '/' + prefix + '.tar'
        print("[INFO] DQN model saved succesfully to ", path)
        torch.save(checkpoint, path)
        
    def run_episode(self):
        state, info = self.env.reset()
        episode_steps = 0
        rewards = 0
        losses = list()
        while True:
            action = self.get_action(state, self.epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_steps += 1
            done = terminated + truncated
            # Save the new transition
            transition = (state, action, reward, next_state, done, info)
            self.add_transition_to_memory(transition)
            if len(self.memory) >= self.hyper_params.update_starts_from:
                if self.total_step % self.hyper_params.train_freq == 0:
                    for _ in range(self.hyper_params.multiple_update):
                        loss = self.update_model()
                        losses.append(loss)
            state = next_state
            rewards += reward
            episode_steps += 1
            self.total_step += 1
            if done:
                break
        return losses, self.episode_step, rewards, self.total_step
    
    def run_game(self):
        self.success = 0
        pbar_dqn = tqdm(total=self.args.iteration_num)
        for i_episode in range(1, self.args.iteration_num + 1):
            self.i_episode = i_episode
            losses, self.episode_step, score, self.total_step = self.run_episode()
            self.do_post_episode_update()
            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, 
                             avg_loss[0], avg_loss[1], 
                             self.epsilon, score)
                wandb.log({
                    "episode" : self.i_episode, 
                    "epsilon" : self.epsilon,
                    "score" : score,
                    "step" : self.total_step
                })
            if i_episode % self.network_cfg.save_freequency == 0:
                params = {
                            "dqn_state_dict": self.dqn.state_dict(),
                            "dqn_target_state_dict": self.dqn_target.state_dict(),
                            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
                        }
                self.save_models(params, self.save_dir, 'dqn_model_{}'.format(i_episode))
            pbar_dqn.update(1)
            pbar_dqn.set_description("Episode {}".format(i_episode))
        pbar_dqn.close()    
                    
    def learn(self):
        self.run_game()
        #Dump the trajectories and model weights
        params = {
        "dqn_state_dict": self.dqn.state_dict(),
        "dqn_target_state_dict": self.dqn_target.state_dict(),
        "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }
        self.save_models(params, self.save_dir, 'dqn_model')
        self.env.close()
        
    def do_post_episode_update(self, *argv):
        if self.total_step >= self.hyper_params.init_random_actions:
            # decrease epsilon
            self.epsilon = max(self.min_epsilon, self.hyper_params.epsilon_decay * self.epsilon)
            
    def do_safety_advise_threshold_update(self):
        if self.total_step >= self.hyper_params.init_random_actions:
            self.safety_advising_threshold = max(self.min_safety_advising_threshold, 
                                                 self.safety_advising_threshold_decay * self.safety_advising_threshold)
            
    def set_episode_step(self):
        self.total_step = self.hyper_params.init_random_actions + 1
            


