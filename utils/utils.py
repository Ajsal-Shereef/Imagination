import os
import cv2
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
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                # Forward pass
                inference_out, reconstruction = model(data)
                labels = torch.argmax(inference_out['prob_cat'], dim=-1)
                all_z.append(inference_out['latent'].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
    if method == 'pca':
        reducer = PCA(n_components=3)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_{checkpoint}.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne_{checkpoint}.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
    fig = plt.figure(figsize=(16,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(reduced_z[:,0], reduced_z[:,1], reduced_z[:,2], c=all_labels, cmap='tab10')
    ax.set_title(f'Latent Space Visualization using {method.upper()}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.grid(True)
    ax.legend()
    fig.savefig(fig_save_name)
    plt.close(fig)
    
    print(f"Latent space visualization saved to {fig_save_name}")
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
    
class TwoListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
    
    def __len__(self):
        return len(self.list1)  # Assume both lists are of the same length
    
    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx]
    
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

def collect_random(env, dataset, num_samples=200):
    state, info = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated + terminated
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, info = env.reset()
            
def write_video(frames, episode, dump_dir, frameSize = (224, 224)):
    os.makedirs(dump_dir, exist_ok=True)
    video = cv2.VideoWriter(dump_dir + '/{}.avi'.format(episode),cv2.VideoWriter_fourcc(*'DIVX'), 1, frameSize, isColor=True)
    for img in frames:
        video.write(img)
    video.release()
            
def test(config, env, agent, save_dir, n_episode=5):
    for i_episode in range(int(n_episode)):
        frame_array = []
        state, info = env.reset()
        frame = env.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_array.append(frame)
        done = False
        score = 0
        episode_step = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_array.append(frame)
            state = next_state

        write_video(frame_array, i_episode, save_dir)