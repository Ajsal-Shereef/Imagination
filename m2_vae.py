import wandb
import json
import hydra
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utils import *
from itertools import cycle
from torch.autograd import Variable
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from omegaconf import DictConfig, OmegaConf
from architectures.m2_vae.variational import SVI
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

def get_vae_gmm_mapping(vae_predictions, gmm_labels):
    # GMM cluster alignment with the VAE
    # Step 1: Aggregate VAE Probabilities for Each GMM Cluster
    n_clusters = 3
    n_classes = vae_predictions.shape[1]

    cluster_probs = np.zeros((n_clusters, n_classes))

    for cluster in range(n_clusters):
        cluster_indices = np.where(gmm_labels == cluster)[0]
        cluster_probs[cluster] = vae_predictions[cluster_indices].mean(axis=0)
        
    # Step 2: Solve the Best Cluster Mapping Using Hungarian Algorithm
    cost_matrix = -cluster_probs  # Use negative because Hungarian minimizes cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Step 3: Align GMM Clusters to VAE Clusters
    mapping = {gmm_cluster: vae_cluster for gmm_cluster, vae_cluster in zip(row_ind, col_ind)}
    return mapping

@hydra.main(version_base=None, config_path="config", config_name="m2")
def main(args: DictConfig) -> None:
    # Initialize dataset and dataloader
    #Loading the dataset
    datasets = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/data.pkl')
    # captions = get_data(f'{args.P_VAE_General.datapath}/{args.P_VAE_General.env}/captions.pkl')
    class_probs = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/class_prob.pkl')
    
    unlabelled_data, labelled_data, _, labels = train_test_split(datasets, class_probs, test_size=0.10, random_state=42)
    print(len(labelled_data))
    # alpha = 0.1 * len(unlabelled_data) / len(labelled_data)
    alpha = 2000
    
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
        iter = 0
        for unlabelled, (labelled, labels) in zip(unlabelled_data_loader, cycle(labelled_data_loader)):
            x, y, u = Variable(labelled).to(device).float(), Variable(labels).to(device).float(), Variable(unlabelled[0]).to(device).float()

            labelled_reconstruction, x_latent, mu, log_var, _ = model(x,y)
            L = model.L(x, y, labelled_reconstruction, mu, log_var)
            
            # dummy_y = torch.full((u.shape[0], 2), 0.5).to(device)
            unlabelled_reconstruction, u_latent, mu, log_var, y_pred_unlabelled = model(u)
            reconstruction_error, kl_loss, U = model.U(u, y_pred_unlabelled, unlabelled_reconstruction, mu, log_var)
            
            #Classification loss for labelled data
            # Regular cross entropy
            y_pred_labelled = model.classify(x_latent)
            classication_loss = torch.sum(y * torch.log(y_pred_labelled + 1e-8), dim=1).mean()
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
            
            # Mask to identify rows where the target is not [0.5, 0.5]
            mask = ~(torch.all(y == 0.5, dim=1))

            # Calculate the predictions and the targets for valid rows
            valid_y_pred = torch.max(y_pred_labelled[mask], 1)[1].data
            valid_y = torch.max(y[mask], 1)[1].data

            # Compute accuracy only for valid rows
            acc = torch.mean((valid_y_pred == valid_y).float())
            accuracy += acc.item()
            iter += 1
        wandb.log({"Loss" : total_loss/len(unlabelled_data_loader),
                   "Classification_loss" : -classication_loss/len(unlabelled_data_loader),
                   "reconstruction_error" : reconstruction_error/len(unlabelled_data_loader),
                   "kl_loss" : kl_loss/len(unlabelled_data_loader),
                   "L loss" : L_losses/len(unlabelled_data_loader),
                   "U loss" : U_losses/len(unlabelled_data_loader),
                   "Accuracy" : accuracy/len(unlabelled_data_loader)}, step=epoch)
        epoch_bar.set_description(f"Accuracy {accuracy/len(unlabelled_data_loader)} Loss {total_loss/len(unlabelled)}")
    model.save("models/m2_vae")
    print("[INFO] M2 model saved to models/m2_vae")
    model.eval()
    # Fit a Gaussian Mixture Model
    full_data = TwoListDataset(datasets, class_probs)
    full_data_loader = DataLoader(full_data, batch_size=900, shuffle=True)
    reduced_data = []
    vae_predictions = []
    for data, true_label in full_data_loader:
        _, img_z, _, _, pred_label = model(data.to(device).float())
        reduced_data.append(img_z.detach().cpu().numpy())
        vae_predictions.append(true_label.detach().cpu().numpy())
    reduced_data = np.concatenate(reduced_data, axis=0)
    vae_predictions = np.concatenate(vae_predictions, axis=0)
    
    gmm = GaussianMixture(n_components=args.M2_General.num_goals + 1, random_state=42)
    gmm.fit(reduced_data)
    labels = gmm.predict(reduced_data)
    centroids = gmm.means_
    
    mapping = get_vae_gmm_mapping(vae_predictions, labels)
    # Prepare JSON Data
    output_data = {}
    for gmm_cluster, vae_cluster in mapping.items():
        output_data[f"Goal_{vae_cluster}"] = centroids[gmm_cluster].tolist()

    # Save JSON to Disk
    output_file = "models/m2_vae/gmm_vae_cluster_mapping.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    #Fit a PCA to further reduce the components for visualizations
    reducer = PCA(n_components=2)
    reduced_z = reducer.fit_transform(reduced_data)
    gmm = GaussianMixture(n_components=args.M2_General.num_goals + 1, random_state=42)
    gmm.fit(reduced_z)
    labels = gmm.predict(reduced_z)
    centroids = gmm.means_
    
    #Plot to visualize the clustering
    data_dir = data_dir = f'visualizations/m2_vae'
    # Define grid for the heatmap
    x = np.linspace(reduced_z[:, 0].min() - 1, reduced_z[:, 0].max() + 1, 300)
    y = np.linspace(reduced_z[:, 1].min() - 1, reduced_z[:, 1].max() + 1, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T

    # Compute probabilities for each point on the grid
    probs = gmm.predict_proba(grid)
    max_probs = probs.max(axis=1).reshape(X_grid.shape)
    
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, args.M2_General.num_goals + 1))

    # Plot the data points, centroids, and heatmap
    plt.figure(figsize=(10, 8))

    # Plot each cluster separately for the legend
    for cluster in range(args.M2_General.num_goals + 1):
        if cluster in mapping.keys():
            legend_label = f'Goal {mapping[cluster]}'
        else:
            legend_label = 'Neutral states'
        plt.scatter(
            reduced_z[labels == cluster, 0], reduced_z[labels == cluster, 1], 
            color=colors[cluster], s=30, alpha=0.6, label=legend_label
        )

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')

    # Add heatmap
    plt.contourf(X_grid, Y_grid, max_probs, levels=20, cmap='coolwarm', alpha=0.20)
    plt.colorbar(label='Cluster Membership Probability')
    
    # Custom legend handles
    # legend_handles = [
    #     Patch(facecolor=colors[0], label='Cluster 1'),
    #     Patch(facecolor=colors[1], label='Cluster 2'),
    #     Patch(facecolor=colors[2], label='Cluster 3'),
    #     Patch(facecolor='red', label='Centroids')
    # ]

    # Add title, labels, legend, and grid
    plt.title('Gaussian Mixture Clustering with Cluster Heatmap')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{data_dir}/Gaussian_Mixture_Clustering.png')  # Save the plot as a file
    
    # Prepare JSON Data
    output_data = {}
    for gmm_cluster, vae_cluster in mapping.items():
        output_data[f"Goal_{vae_cluster}"] = centroids[gmm_cluster].tolist()

    # Save JSON to Disk
    output_file = "models/m2_vae/gmm_vae_cluster_mapping_reduced.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
if __name__ == "__main__":
    wandb.init(project="M2_training")
    main()        
        
        