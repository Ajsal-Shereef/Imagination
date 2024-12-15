import wandb
import hydra
import torch

from tqdm import tqdm
from utils.utils import *
from itertools import cycle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from architectures.m2_vae.variational import SVI
from sklearn.model_selection import train_test_split
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

@hydra.main(version_base=None, config_path="config", config_name="m2")
def main(args: DictConfig) -> None:
    # Initialize dataset and dataloader
    #Loading the dataset
    datasets = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/data.pkl')
    # captions = get_data(f'{args.P_VAE_General.datapath}/{args.P_VAE_General.env}/captions.pkl')
    class_probs = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/class_prob.pkl')
    
    unlabelled_data, labelled_data, _, labels = train_test_split(datasets, class_probs, test_size=0.05, random_state=42)
    print(len(labelled_data))
    # alpha = 0.1 * len(unlabelled_data) / len(labelled_data)
    alpha = 100
    
    unlabelled_data = Dataset(unlabelled_data)
    labelled_data = TwoListDataset(labelled_data, labels)
    unlabelled_data_loader = DataLoader(unlabelled_data, batch_size=900, shuffle=True)
    labelled_data_loader = DataLoader(labelled_data, batch_size=900, shuffle=True)
    #Creating the model and the optimizer
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.num_goals, args.M2_Network.latent_dim, args.M2_Network.encoder_hidden_dim]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.M2_Network.lr_model, betas=(0.9, 0.999))
    
    wandb.watch(model)
    epoch_bar = tqdm(range(args.M2_Network.epochs), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_loss, accuracy = (0, 0)
        classification_losses, reconstruction_errors, kl_losses  = 0, 0, 0
        L_losses, U_losses = 0, 0
        for unlabelled, (labelled, labels) in zip(unlabelled_data_loader, cycle(labelled_data_loader)):
            x, y, u = Variable(labelled).to(device).float(), Variable(labels).to(device).float(), Variable(unlabelled[0]).to(device).float()

            labelled_reconstruction, latent, mu, log_var = model(x,y)
            L = model.L(x, y, labelled_reconstruction, mu, log_var)
            
            # Add auxiliary classification loss q(y|x)
            logits = model.classify(u)
            # dummy_y = torch.full((u.shape[0], 2), 0.5).to(device)
            unlabelled_reconstruction, latent, mu, log_var = model(u, logits)
            reconstruction_error, kl_loss, U = model.U(u, logits, unlabelled_reconstruction, mu, log_var)
            
            #Classification loss for labelled data
            logits = model.classify(x)
            # Regular cross entropy
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
            #The negative sign infront of L andU is as in the paper. model.U calculate the U.
            J_alpha = -L - alpha * classication_loss - U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()
            classification_losses += classication_loss.item()
            reconstruction_errors += reconstruction_error
            kl_losses += kl_loss
            L_losses += -L.item()
            U_losses += -U.item()
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())
        wandb.log({"Loss" : total_loss/len(unlabelled),
                   "Classification_loss" : -classication_loss/len(unlabelled),
                   "reconstruction_error" : reconstruction_error/len(unlabelled),
                   "kl_loss" : kl_loss/len(unlabelled),
                   "L loss" : L_losses/len(unlabelled),
                   "U loss" : U_losses/len(unlabelled)}, step=epoch)
        epoch_bar.set_description(f"Accuracy {accuracy} Loss {total_loss/len(unlabelled)}")
    model.save("models/m2_vae")
if __name__ == "__main__":
    wandb.init(project="M2_training")
    main()        
        
        