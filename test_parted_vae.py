import os
import hydra

from utils.utils import *
from partedvae.models import VAE
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from architectures.m2_vae.dgm import DeepGenerativeModel
from utils.collect_vae_training_data import collect_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_latent_space(model, dataloader, device, all_z=[], all_labels=[], method='pca', save_path=''):
    """
    Visualize the latent space using PCA or t-SNE.

    Args:
        model (VAE): Trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        method (str): 'pca' or 'tsne'.
        save_path (str): Path to save the visualization plot.
    """
    model.eval()
    if len(all_z) == 0:
        all_z = []
        all_labels = []
        true_labels = []
        with torch.no_grad():
            for data, true_label in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                true_label = true_label.float().to(device)
                true_labels.append(true_label.detach().cpu().numpy())
                # Forward pass
                x_mu, z, z_mu, z_log_var = model(data, true_label)
                labels = model.classify(data)
                latent = z
                all_z.append(latent.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        
    fig_save_name = f'{save_path}/latent_space_.png'
    true_fig_save_name = f'{save_path}/latent_space_true_label.png'

    # plt.scatter(all_z[:,0], all_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='Probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(fig_save_name)
    # plt.close()
    
    # plt.scatter(all_z[:,0], all_z[:,1], c=true_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='True probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(true_fig_save_name)
    # plt.close()
    
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
    plt.scatter(reduced_z[:,0], reduced_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Predicted probability of Class 1')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.savefig(fig_save_name)
    plt.close()
    
    plt.scatter(reduced_z[:,0], reduced_z[:,1], c=true_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(label='True probability of Class 1')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.savefig(true_fig_save_name)
    plt.close()
    
    
    print(f"Latent space visualization saved to {fig_save_name}")
    return all_z, all_labels

@hydra.main(version_base=None, config_path="config", config_name="m2")
def main(args: DictConfig) -> None:
    #Loading the dataset
    if args.M2_General.env == "SimplePickup":
        from env.env import SimplePickup #TransitionCaptioner
        env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
        
    datasets, _, class_probs = collect_data(env, True, 100, 20, device)
    
    # Initialize dataset and dataloader
    dataloader = TwoListDataset(datasets, class_probs)
    train_loader = DataLoader(dataloader, batch_size=900, shuffle=True)
    
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.num_goals, args.M2_Network.latent_dim, args.M2_Network.encoder_hidden_dim]).to(device)
    model.load("models/m2_vae/m2_vae.pth")
    model.to(device)
    model.eval()
    
    # Visualization of the latent space 
    data_dir = f'visualizations/m2_vae'
    os.makedirs(data_dir, exist_ok=True)
    latent, labels = visualize_latent_space(model, train_loader, device, method='pca', save_path=data_dir)
    # latent = visualize_latent_space(model, dataloader, device, latent, labels, method='tsne', save_path=data_dir)
    
if __name__ == "__main__":
    main()