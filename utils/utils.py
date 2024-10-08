import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset


def visualize_latent_space(model, dataloader, device, all_z=[], method='pca', save_path='latent_space.png'):
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
            for sentences in tqdm(dataloader, desc="Collecting Latent Vectors"):
                sentences = list(sentences)
                z, _, _ = model.reparameterize(*model.encode(sentences))
                all_z.append(z.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]

    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=40, n_iter=1000)
        reduced_z = reducer.fit_transform(all_z)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_z[:, 0], reduced_z[:, 1], alpha=0.7)
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Latent space visualization saved to {save_path}")
    return all_z

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