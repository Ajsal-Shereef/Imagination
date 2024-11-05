import os
import wandb
import hydra
import torch
import itertools
from torch import optim
from utils.utils import *
from partedvae.models import VAE
from partedvae.training import Trainer
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(version_base=None, config_path="config", config_name="parted_vae")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    #Loading the dataset
    datasets = get_data(f'{args.General.datapath}/{args.General.env}/data.pkl')
    captions = get_data(f'{args.General.datapath}/{args.General.env}/captions.pkl')
    class_probs = get_data(f'{args.General.datapath}/{args.General.env}/class_prob.pkl')
    
    dataset, labelled_data, unused_label, labels = train_test_split(datasets, class_probs, test_size=0.01, random_state=42)
    
    # Initialize dataset and dataloader
    dataloader = TwoListDataset(dataset, unused_label)
    train_loader = DataLoader(dataloader, batch_size=900, shuffle=True)
    
    dataloader = TwoListDataset(labelled_data, labels)
    warm_up_loader = DataLoader(dataloader, batch_size=100, shuffle=True)

    disc_priors = [[1/args.General.num_goals]*args.General.num_goals]
    latent_spec = args.Training.latent_spec
    z_capacity = args.Training.z_capacity
    u_capacity = args.Training.u_capacity
    g_c = args.Training.g_c
    g_h = args.Training.g_h
    g_bc = args.Training.g_bc
    bc_threshold = args.Training.bc_threshold
    recon_type = args.Training.recon_loss
    epochs = args.Training.epochs
    
    save_dir = f'{args.General.load_model_path}/parted_vae'
    os.makedirs(save_dir, exist_ok=True)
    
    model = VAE(args.Training.input_dim, 
                args.Training.encoder_hidden_dim, 
                args.Training.decoder_hidden_dim, 
                args.Training.output_size, 
                latent_spec, 
                c_priors=disc_priors,
                save_dir = save_dir,
                device=device)
    
    if args.General.LOAD_MODEL:
        # Note: When you load a model, capacities are restarted, which isn't intuitive if you are gonna re-train it
        model = model.load(save_dir, device=device)
        model.sigmoid_coef = 8.
 
    if args.General.TRAIN:
        optimizer_warm_up = optim.Adam(itertools.chain(*[
            model.encoder.parameters(),
            model.h_to_c_logit_fc.parameters()
        ]), lr=args.Training.lr_warm_up)
        optimizer_model = optim.Adam(model.parameters(), lr=args.Training.lr_model)
        optimizers = [optimizer_warm_up, optimizer_model]

        trainer = Trainer(model, optimizers, device=device, recon_type=recon_type,
                          z_capacity=z_capacity, u_capacity=u_capacity, c_gamma=g_c, entropy_gamma=g_h,
                          bc_gamma=g_bc, bc_threshold=bc_threshold, save_freequency = args.Training.save_freequency,
                          model_save_path = f'{args.General.load_model_path}/parted_vae')
        trainer.train(train_loader, warm_up_loader=warm_up_loader, epochs=epochs, run_after_epoch=None,
                      run_after_epoch_args=[])

    
if __name__ == "__main__":
    wandb.init(project="Imagination-Parted VAE_training")
    main()   
    
