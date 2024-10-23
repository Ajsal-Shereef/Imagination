import torch
from vae.vae import GMVAE
from env.env import SimplePickup
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer

vae_path = "models/all-MiniLM-L6-v2/Feature_based/vae_normal.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
goal_gen = GetLLMGoals()
goals = goal_gen.llmGoals([])
# Load the sentecebert model to get the embedding of the goals from the LLM
sentencebert = SentenceTransformer("all-MiniLM-L6-v2")
# Define prior means (mu_p) for each mixture component as the output from the sentencebert model
mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
# Define prior means (mu_p) for each mixture component
# Initialize VAE with learnable prior means
latent_dim = sentencebert.get_sentence_embedding_dimension()
num_mixtures = len(goals)

vae = GMVAE(
        input_dim = 504, 
        encoder_hidden = [1024,1024,512,512,512,256,256,256,256], #Don't forget to edit the snippet above as well
        decoder_hidden = [256,256,256,256,512,512,512,1024,1024], 
        latent_dim=latent_dim, 
        num_mixtures=num_mixtures, 
        mu_p=mu_p
    )
vae.load(vae_path)
vae.to(device)

env = SimplePickup(max_steps=20, agent_view_size=5, size=7)

total_step = 0
while total_step < 20000:
    state, info = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_step += 1
        vae_input = torch.tensor(next_state).unsqueeze(0).float().to(device)
        vae_output = vae(vae_input)
        cluster_prob = vae_output[0]['categorical']
        if vae_output[0]['categorical'][0][0] > 0.90:
            print("First gaussian is greater than 90%")
            print(env.extract_objects_from_observation())
        elif vae_output[0]['categorical'][0][1] > 0.90:
            print("Second goals is greater than 90%")
            print(env.extract_objects_from_observation())
        done = terminated + truncated
        state = next_state
        if done:
            break

