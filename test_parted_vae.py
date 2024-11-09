import os
import hydra

from utils.utils import *
from partedvae.models import VAE
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

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
        with torch.no_grad():
            for data, label in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                # Forward pass
                reconstruction, inference_out = model(data)
                labels = torch.exp(inference_out['log_c'])
                latent = inference_out['u'][0]
                all_z.append(latent.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
        
    fig_save_name = f'{save_path}/latent_space_.png'

    plt.scatter(all_z[:,0], all_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Probability of Class 1')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.savefig(fig_save_name)
    plt.close()
    
    
    # if method == 'pca':
    #     reducer = PCA(n_components=3)
    #     reduced_z = reducer.fit_transform(all_z)
    #     fig_save_name = f'{save_path}/latent_space_pca_.png'
    # elif method == 'tsne':
    #     reducer = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    #     reduced_z = reducer.fit_transform(all_z)
    #     fig_save_name = f'{save_path}/latent_space_tsne.png'
    # else:
    #     raise ValueError("Method must be 'pca' or 'tsne'.")
    
    # fig = plt.figure(figsize=(16,10))
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(reduced_z[:,0], reduced_z[:,1], reduced_z[:,2], c=all_labels, cmap='tab10')
    # ax.set_title(f'Latent Space Visualization using {method.upper()}')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # ax.grid(True)
    # ax.legend()
    # fig.savefig(fig_save_name)
    # plt.close(fig)
    
    print(f"Latent space visualization saved to {fig_save_name}")
    return all_z, all_labels

@hydra.main(version_base=None, config_path="config", config_name="parted_vae")
def main(args: DictConfig) -> None:
    #Loading the dataset
    datasets = get_data(f'{args.General.datapath}/{args.General.env}/data.pkl')
    # captions = get_data(f'{args.General.datapath}/{args.General.env}/captions.pkl')
    class_probs = get_data(f'{args.General.datapath}/{args.General.env}/class_prob.pkl')
    
    # Initialize dataset and dataloader
    dataloader = TwoListDataset(datasets, class_probs)
    train_loader = DataLoader(dataloader, batch_size=900, shuffle=True)
    
    latent_spec = args.Training.latent_spec
    disc_priors = [[1/args.General.num_goals]*args.General.num_goals]
    save_dir = os.makedirs(f'{args.General.load_model_path}/parted_vae', exist_ok=True)
    
    model = VAE(args.Training.input_dim, 
                args.Training.encoder_hidden_dim, 
                args.Training.decoder_hidden_dim, 
                args.Training.output_size, 
                latent_spec, 
                c_priors=disc_priors, 
                save_dir=save_dir,
                device=device)
    model.load("models/parted_vae/parted_vae_55000.pth")
    model.to(device)
    model.eval()
    
    # Visualization of the latent space 
    data_dir = f'visualizations/{args.General.env}/Feature_based'
    os.makedirs(data_dir, exist_ok=True)
    latent, labels = visualize_latent_space(model, train_loader, device, method='pca', save_path=data_dir)
    # latent = visualize_latent_space(model, dataloader, device, latent, labels, method='tsne', save_path=data_dir)
    
if __name__ == "__main__":
    main()