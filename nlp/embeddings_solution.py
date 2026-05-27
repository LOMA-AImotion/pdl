"""
This tutorial is based on https://github.com/StatQuest/word_embedding_with_pytorch_and_lightning/blob/main/statquest_word_embedding_with_pytorch_lightning_v2.ipynb
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.patches import Patch


class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        
        super().__init__()
        
        L.seed_everything(seed=42)
        
        self.input_to_hidden = nn.Linear(in_features=7, out_features=3)
        self.hidden_to_output = nn.Linear(in_features=3, out_features=7)
        
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, input): 
        
        hidden = self.input_to_hidden(input)
        hidden = F.relu(hidden)
        
        output_values = self.hidden_to_output(hidden)
                
        return(output_values)
        
        
    def configure_optimizers(self): 
        # this configures the optimizer we want to use for backpropagation.
        
        return Adam(self.parameters(), lr=0.001)

    
    def training_step(self, batch, batch_idx): 
        # take a step during gradient descent.
        
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i) # run input through the neural network
        loss = self.loss(output_i, label_i) ## loss = cross entropy
                    
        return loss

def visualize_embeddings(model, vocab, vocabulary_words=None):
    """
    Visualizes the learned word embeddings in 3D space.
    
    Args:
        model: Trained WordEmbeddingWithLinear model
        vocab: Tensor of one-hot encoded vocabulary
        vocabulary_words: List of vocabulary words to label the embeddings
                         Defaults to the words in our dataset
    """
    if vocabulary_words is None:
        vocabulary_words = ['THI', 'is', 'great', 'PDL', 'at', 'interesting', '<EOS>']
    
    # Get embeddings by passing each one-hot encoded word through input_to_hidden layer
    model.eval()
    with torch.no_grad():
        embeddings = model.input_to_hidden(vocab).numpy()
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colormap and plot each embedding as a point
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(embeddings))]
    
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                        s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Create legend using proxy artists with colors
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=vocabulary_words[i]) 
                      for i in range(len(vocabulary_words))]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Embedding Dimension 1', fontsize=11)
    ax.set_ylabel('Embedding Dimension 2', fontsize=11)
    ax.set_zlabel('Embedding Dimension 3', fontsize=11)
    ax.set_title('Learned Word Embeddings (3D)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    model.train()



# Our dataset: "THI is great" <EOS>; "PDL at THI is interesting" <EOS>
# Vocabulary with <EOS> token added
vocab = torch.tensor([[1., 0., 0., 0., 0., 0., 0.], # one-hot-encoding for THI 
                       [0., 1., 0., 0., 0., 0., 0.], # ...is
                       [0., 0., 1., 0., 0., 0., 0.], # ...great 
                       [0., 0., 0., 1., 0., 0., 0.], # ...PDL 
                       [0., 0., 0., 0., 1., 0., 0.], # ...at  
                       [0., 0., 0., 0., 0., 1., 0.], # ...interesting
                       [0., 0., 0., 0., 0., 0., 1.]]) # ...<EOS>

# Input-label pairs from both sentences:
# Sentence 1: "THI is great" <EOS>
#   THI (0) → is (1)
#   is (1) → great (2)
#   great (2) → <EOS> (6)

# Sentence 2: "PDL at THI is interesting" <EOS>
#   PDL (3) → at (4)
#   at (4) → THI (0)
#   THI (0) → is (1)
#   is (1) → interesting (5)
#   interesting (5) → <EOS> (6)

# Build inputs by indexing into vocab: [THI, is, great, PDL, at, THI, is, interesting]
inputs = vocab[[0, 1, 2, 3, 4, 0, 1, 5]]

labels = vocab[[1, 2, 6, 4, 0, 1, 5, 6]] # Next word after each input: [is, great, <EOS>, at, THI, is, interesting, <EOS>]

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)


embedding_model = WordEmbeddingWithLinear()

visualize_embeddings(embedding_model, vocab)

trainer = L.Trainer(max_epochs=1000)
trainer.fit(embedding_model, train_dataloaders=dataloader)



# Visualize the learned embeddings
visualize_embeddings(embedding_model, vocab)

