import torch
import torch.nn.functional as F
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer, CrossEncoder

vae_path = "models/all-MiniLM-L6-v2/Feature_based/vae_normal.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
scores = model.predict([("The agent ignored the green ball.", "Pick green ball"),
                        ("The agent ignored the green ball.", "Pick purple key")])
print(scores)

# Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
goal_gen = GetLLMGoals()
goals = goal_gen.llmGoals([])
# Load the sentecebert model to get the embedding of the goals from the LLM
sentencebert = SentenceTransformer("all-MiniLM-L12-v2")
# Define prior means (mu_p) for each mixture component as the output from the sentencebert model
mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)

caption1 = 'The agent moved towards to the green ball.'
caption2 = 'The agent moved towards to the purple key.'
caption3 = 'The agent moved equally towards red and green ball'
caption4 = 'The agent ignored the green ball.'
caption5 = 'The agent ignored the purple key.'
caption6 = 'The agent sees green ball and purple key in the current view. Agent is closer to the purple key.'
caption7 = 'The purple key appeared in the current view. Agent is closer to the purple key.'
caption8 = 'The purple key appeared in the current view. Agent is closer to the green ball.'
caption9 = 'The green ball appeared in the current view. Agent is closer to the green ball.'

captions = [caption1, caption2, caption3, caption4, caption5, caption6, caption7]
print(sentencebert.similarity(sentencebert.encode(captions, convert_to_tensor=True, device=device),
                        sentencebert.encode(captions, convert_to_tensor=True, device=device)))

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
cs6_1 = F.cosine_similarity(sentencebert.encode(caption6, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs6_2 = F.cosine_similarity(sentencebert.encode(caption6, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm6 = F.softmax(torch.stack((cs6_1, cs6_2))/0.1, dim=-1)
cs7_1 = F.cosine_similarity(sentencebert.encode(caption7, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs7_2 = F.cosine_similarity(sentencebert.encode(caption7, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm7 = F.softmax(torch.stack((cs7_1, cs7_2))/0.1, dim=-1)
cs8_1 = F.cosine_similarity(sentencebert.encode(caption8, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs8_2 = F.cosine_similarity(sentencebert.encode(caption8, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm8 = F.softmax(torch.stack((cs8_1, cs8_2))/0.1, dim=-1)
cs9_1 = F.cosine_similarity(sentencebert.encode(caption9, convert_to_tensor=True, device=device), mu_p[0], dim=-1)
cs9_2 = F.cosine_similarity(sentencebert.encode(caption9, convert_to_tensor=True, device=device), mu_p[1], dim=-1)
sm9 = F.softmax(torch.stack((cs9_1, cs9_2))/0.1, dim=-1)

print(cs1_1)
print(cs1_2)
print(sm1)
print('1************')
print(cs2_1)
print(cs2_2)
print(sm2)
print('2************')
print(cs3_1)
print(cs3_2)
print(sm3)
print('3************')
print(cs4_1)
print(cs4_2)
print(sm4)
print('4************')
print(cs5_1)
print(cs5_2)
print(sm5)
print('5************')
print(cs6_1)
print(cs6_2)
print(sm6)
print('6************')
print(cs7_1)
print(cs7_2)
print(sm7)
print('7************')
print(cs8_1)
print(cs8_2)
print(sm8)
print('8************')
print(cs9_1)
print(cs9_2)
print(sm9)
print("done")