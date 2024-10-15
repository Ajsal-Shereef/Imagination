import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset


def visualize_latent_space(model, dataloader, device, all_z=[], all_labels=[], method='pca', save_path='', checkpoint=''):
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
        all_data = []
        with torch.no_grad():
            for sentences in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                sentences = sentences.float().to(device)
                mu, logvar, weights = model.encode(sentences)
                labels = torch.where(weights[:, 0] > 0.5, 1, 2)
                z, _, _ = model.reparameterize(mu, logvar, weights)
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_data.append(sentences.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
        all_data = np.concatenate(all_data, axis=0)
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_{checkpoint}.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=40, n_iter=1000)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne_{checkpoint}.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    plt.figure(figsize=(8, 6))
    for label in np.unique(all_labels):
        plt.scatter(reduced_z[all_labels == label, 0], reduced_z[all_labels == label, 1], label=f'Goal {label}', alpha=0.6)
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.legend()
    plt.savefig(fig_save_name)
    plt.close()
    
    # all_data = (all_data-np.mean(all_data))/np.std(all_data)
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_data)
        fig_save_name = f'{save_path}/data_space_pca_{checkpoint}.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=40, n_iter=1000)
        reduced_z = reducer.fit_transform(all_data)
        fig_save_name = f'{save_path}/latent_space_tsne_{checkpoint}.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    plt.figure(figsize=(8, 6))
    for label in np.unique(all_labels):
        plt.scatter(reduced_z[all_labels == label, 0], reduced_z[all_labels == label, 1], label=f'Goal {label}', alpha=0.6)
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig(fig_save_name)
    
    print(f"Latent space visualization saved to {save_path}")
    return all_z, all_labels

def get_data(datapath):
    return np.load(datapath, allow_pickle=True)

class TextDataset(Dataset):
    def __init__(self, sentences):
        """
        Custom Dataset for VAE training.

        Args:
            sentences (list of str): List of sentences.
        """
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Return the sentence at index idx.

        Args:
            idx (int): Index.

        Returns:
            str: Sentence.
        """
        return self.sentences[idx]
    
def normalize_data(data):
    """
    Normalize the data using Z-score normalization (mean = 0, std = 1).
    
    Args:
        data (np.array): The data to normalize (N x features).
        
    Returns:
        np.array: The normalized data.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-7)  # Small epsilon to prevent division by zero
    return normalized_data

def monte_carlo_entropy(alpha, mu, sigma, num_samples=10000):
    """
    Monte Carlo estimation of entropy for a mixture of Gaussians.
    
    Args:
        alpha (list): Weights of the Gaussian components. Must sum to 1.
        mu (list): Means of the Gaussian components.
        sigma (list): Standard deviations of the Gaussian components.
        num_samples (int): Number of samples to generate for Monte Carlo estimation.

    Returns:
        float: Estimated entropy of the mixture distribution.
    """
    M = len(alpha)  # Number of components in the mixture

    # Step 1: Sample component indices according to the mixture weights alpha
    component_indices = np.random.choice(M, size=num_samples, p=alpha)

    # Step 2: Generate samples from the corresponding Gaussians
    samples = np.array([np.random.normal(mu[i], sigma[i]) for i in component_indices])

    # Step 3: Compute the density of each sample under the full mixture distribution
    mixture_density = np.zeros(num_samples)
    for i in range(M):
        # Compute the contribution of the i-th Gaussian to the density of each sample
        component_density = alpha[i] * norm.pdf(samples, mu[i], sigma[i])
        mixture_density += component_density

    # Step 4: Compute the entropy using the Monte Carlo estimate
    entropy = -np.mean(np.log(mixture_density + 1e-10))  # Adding a small constant for numerical stability

    return entropy