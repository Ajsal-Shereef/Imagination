import torch
import pickle
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader


# Load your feature and caption data
with open('data/all-MiniLM-L12-v2/captions.pkl', 'rb') as f:
    features = pickle.load(f)

with open('data/all-MiniLM-L12-v2/data.pkl', 'rb') as f:
    captions = pickle.load(f)

# Define a function to create training examples
def create_training_examples(features, captions):
    examples = []
    for feature, caption in zip(features, captions):
        examples.append(InputExample(texts=[caption], label=[feature]))
    return examples

# Create training examples
train_examples = create_training_examples(features, captions)

# Create a Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Create a training dataset
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16) 


# Define the training loss
train_loss = losses.ContrastiveLoss(model=model)

# Train the model
model.fit(train_dataloader, epochs=3, loss=train_loss)

# Save the fine-tuned model
model.save('fine_tuned_model')