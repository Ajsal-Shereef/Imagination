import torch
import torch.nn.functional as F
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer

vae_path = "models/all-MiniLM-L6-v2/Feature_based/vae_normal.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
goal_gen = GetLLMGoals()
goals = goal_gen.llmGoals([])
# Load the sentecebert model to get the embedding of the goals from the LLM
sentencebert = SentenceTransformer("paraphrase-albert-small-v2")
# Define prior means (mu_p) for each mixture component as the output from the sentencebert model
mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)

caption1 = 'You see red ball, green ball'
caption2 = 'You see red ball'
caption3 = 'You see green ball'
caption4 = 'You see nothing'
caption5 = 'You see green ball, red ball'

cs1_1 = F.cosine_similarity(sentencebert.encode(caption1, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs1_2 = F.cosine_similarity(sentencebert.encode(caption1, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm1 = F.softmax(torch.stack((cs1_1, cs1_2))/0.1, dim=-1)
cs2_1 = F.cosine_similarity(sentencebert.encode(caption2, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs2_2 = F.cosine_similarity(sentencebert.encode(caption2, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm2 = F.softmax(torch.stack((cs2_1, cs2_2))/0.1, dim=-1)
cs3_1 = F.cosine_similarity(sentencebert.encode(caption3, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs3_2 = F.cosine_similarity(sentencebert.encode(caption3, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm3 = F.softmax(torch.stack((cs3_1, cs3_2))/0.1, dim=-1)
cs4_1 = F.cosine_similarity(sentencebert.encode(caption4, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs4_2 = F.cosine_similarity(sentencebert.encode(caption4, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm4 = F.softmax(torch.stack((cs4_1, cs4_2))/0.1, dim=-1)
cs5_1 = F.cosine_similarity(sentencebert.encode(caption5, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs5_2 = F.cosine_similarity(sentencebert.encode(caption5, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm5 = F.softmax(torch.stack((cs5_1, cs5_2))/0.1, dim=-1)

print(cs1_1)
print(cs1_2)
print(sm1)
print('************')
print(cs2_1)
print(cs2_2)
print(sm2)
print('************')
print(cs3_1)
print(cs3_2)
print(sm3)
print('************')
print(cs4_1)
print(cs4_2)
print(sm4)
print('************')
print(cs5_1)
print(cs5_2)
print(sm5)
print("done")