import os
import wandb
import hydra
import torch
import itertools
from torch import optim
from helper_functions.utils import *
from dqn.dqn import DQNAgent
from sac_agent.agent import SAC
from partedvae.models import VAE
from partedvae.training import Trainer
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.General.max_ep_len, agent_view_size=5, size=7)
    
    #Loading the dataset
    datasets = get_data(f'{args.P_VAE_General.datapath}/{args.P_VAE_General.env}/data.pkl')
    # captions = get_data(f'{args.P_VAE_General.datapath}/{args.P_VAE_General.env}/captions.pkl')
    class_probs = get_data(f'{args.P_VAE_General.datapath}/{args.P_VAE_General.env}/class_prob.pkl')
    
    dataset, labelled_data, unused_label, labels = train_test_split(datasets, class_probs, test_size=0.023, random_state=42)
    print("[INFO] Number of labelled data: ", len(labelled_data))
    
    # Initialize dataset and dataloader
    dataloader = TwoListDataset(dataset, unused_label)
    train_loader = DataLoader(dataloader, batch_size=900, shuffle=True)
    
    dataloader = TwoListDataset(labelled_data, labels)
    warm_up_loader = DataLoader(dataloader, batch_size=100, shuffle=True)

    disc_priors = [[1/args.P_VAE_General.num_goals]*args.P_VAE_General.num_goals]
    latent_spec = args.P_VAE_Network.latent_spec
    z_capacity = args.P_VAE_Network.z_capacity
    u_capacity = args.P_VAE_Network.u_capacity
    g_c = args.P_VAE_Network.g_c
    g_h = args.P_VAE_Network.g_h
    g_bc = args.P_VAE_Network.g_bc
    bc_threshold = args.P_VAE_Network.bc_threshold
    recon_type = args.P_VAE_Network.recon_loss
    epochs = args.P_VAE_Network.epochs
    
    save_dir = f'{args.P_VAE_General.load_model_path}/parted_vae'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.Imagination_General.agent == 'dqn':
        agent = DQNAgent(env, 
                        args.General, 
                        args.policy_config, 
                        args.policy_network_cfg, 
                        args.policy_network_cfg, '')
    else:
        agent = SAC(args,
                    state_size = env.observation_space.shape[0],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.memory_size)
    agent.to(device)
    agent.load_params(args.Imagination_General.agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
    
    model = VAE(args.P_VAE_Network.input_dim, 
                args.P_VAE_Network.encoder_hidden_dim, 
                args.P_VAE_Network.decoder_hidden_dim, 
                args.P_VAE_Network.output_size, 
                latent_spec, 
                c_priors=disc_priors,
                save_dir = save_dir,
                imagination_net_config = args.Imagination_Network,
                env = env,
                device=device)
    
    wandb.watch(model)

    if args.P_VAE_General.LOAD_MODEL:
        # Note: When you load a model, capacities are restarted, which isn't intuitive if you are gonna re-train it
        model = model.load(save_dir, device=device)
        model.sigmoid_coef = 8.
 
    if args.P_VAE_General.TRAIN:
        optimizer_warm_up = optim.Adam(itertools.chain(*[
            model.encoder.parameters(),
            model.h_to_c_logit_fc.parameters()
        ]), lr=args.P_VAE_Network.lr_warm_up)
        optimizer_imagination_net = optim.Adam(itertools.chain(*[
            model.encoder.parameters(),
            model.imagination_net.parameters()
        ]), lr=args.P_VAE_Network.lr_warm_up)
        # Get all model parameters
        all_params = set(model.parameters())
        # Get parameters to exclude
        excluded_params = set(model.imagination_net.parameters())
        
        # Define the parameters for the optimizer, excluding the chosen network
        params_to_optimize = all_params - excluded_params
        optimizer_model = optim.Adam(params_to_optimize, lr=args.P_VAE_Network.lr_model)
        optimizers = [optimizer_warm_up, optimizer_imagination_net, optimizer_model]

        trainer = Trainer(model, optimizers, agent=agent, device=device, recon_type=recon_type,
                          z_capacity=z_capacity, u_capacity=u_capacity, c_gamma=g_c, entropy_gamma=g_h,
                          bc_gamma=g_bc, bc_threshold=bc_threshold, save_freequency = args.P_VAE_Network.save_freequency,
                          model_save_path = f'{args.P_VAE_General.load_model_path}/parted_vae')
        trainer.train(train_loader, warm_up_loader=warm_up_loader, epochs=epochs, run_after_epoch=None,
                      run_after_epoch_args=[])

    
if __name__ == "__main__":
    wandb.init(project="Imagination-Parted VAE_training")
    main()   
    
