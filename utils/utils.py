import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from torch.autograd import Variable


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
            for data, indices in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                # Forward pass
                reconstruction, inference_out = model(data)
                # labels = torch.argmax(inference_out['prob_cat'], dim=-1)
                labels = inference_out['prob_cat']
                all_z.append(inference_out['latent'].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_{checkpoint}.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne_{checkpoint}.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
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
    
    plt.scatter(all_z[:,0], all_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Probability of Class 1')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.savefig(fig_save_name)
    plt.close()
    
    print(f"Latent space visualization saved to {fig_save_name}")
    return all_z, all_labels

def get_data(datapath):
    return np.load(datapath, allow_pickle=True)

class Dataset(Dataset):
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
        return self.sentences[idx], idx
    
class TwoListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
    
    def __len__(self):
        return len(self.list1)  # Assume both lists are of the same length
    
    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx]
    
class ThreeListDataset(Dataset):
    def __init__(self, list1, list2, list3):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
    
    def __len__(self):
        return len(self.list1)  # Assume both lists are of the same length
    
    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx], self.list3[idx]
    
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
            
def test(env, agent, save_dir, n_episode=5):
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
        
def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max