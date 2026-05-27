"""
This tutorial is based on https://github.com/StatQuest/word_embedding_with_pytorch_and_lightning/blob/main/statquest_word_embedding_with_pytorch_lightning_v2.ipynb
"""

import torch 
import torch.nn as nn 
import lightning as L
from torch.optim import Adam 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.patches import Patch


class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):
        
        super().__init__()
        
        # TODO define layers 
        
        # We'll use CrossEntropyLoss in training_step()
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, input): 
        # TODO  
        pass
        
        
    def configure_optimizers(self): 
        # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr=0.001)

    
    def training_step(self, batch, batch_idx): 
        # take a step during gradient descent.
        
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i) # run input through the neural network
        loss = self.loss(output_i, label_i) ## loss = cross entropy
                    
        return loss

def visualize_embeddings(model, vocab, vocabulary_words):
    """
    Visualizes the learned word embeddings in 3D space.
    
    Args:
        model: Trained WordEmbeddingWithLinear model
        vocab: Tensor of one-hot encoded vocabulary
        vocabulary_words: List of vocabulary words to label the embeddings
                         Defaults to the words in our dataset
    """
    
    # Get embeddings by passing each one-hot encoded word through input_to_hidden layer
    model.eval()
    with torch.no_grad():
        embeddings = model.input_to_hidden(vocab).numpy()
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colormap and plot each embedding as a point
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (len(embeddings) - 1)) for i in range(len(embeddings))]
    
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


# TODO define dataset as one-hot vectors here. How can we separate sentences?
vocab = None

# Build your input-label pairs, i.e., which word follows a particular word from our dataset (the sentences built by our vocab).: 
# inputs = vocab[[0, 1, 2, 3, 4, 0, 1, 5]]
# labels = vocab[[1, 2, 6, 4, 0, 1, 5, 6]] # Next word after each input

inputs = None
labels = None

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)


embedding_model = WordEmbeddingWithLinear()

visualize_embeddings(embedding_model, vocab)

trainer = L.Trainer(max_epochs=1000)
trainer.fit(embedding_model, train_dataloaders=dataloader)



# Visualize the learned embeddings
visualize_embeddings(embedding_model, vocab)

