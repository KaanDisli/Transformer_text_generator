import numpy as np
import random  
import torch
import torch.nn as nn
with open("Shakespeare_text.txt") as file:
    text = file.read()



learning_rate = 3e-4
text_size = len(text)
batch_size = 64
context_size = 256
dim = 2
chars = sorted(list(set(text)))
vocab_size = len(chars)
training_text  = text[:text_size]
training_size = len(training_text)
val_text = text[training_size:]
val_size = len(val_text)
embd_size =384
head_size = 64
n_layer = 6 
dropout  = 0.3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[char] for char in s ]
decode = lambda s: ''.join([itos[i] for i in s])


#generates a random batch of characters with (context_size, batch_size)
def get_batch():
    ix = torch.randint(training_size - context_size, (batch_size,))
    x = [training_text[i:i+context_size] for i in ix]
    x = [encode(char) for char  in x]
    x = torch.tensor(x)
    label = [training_text[i+1:i+context_size+1] for i in ix]
    label = [encode(char) for char in label ] 
    label = torch.tensor(label)
    return x , label


def get_validation_batch():
    ix = torch.randint(val_size - context_size, (batch_size,))
    x = [val_text[i:i+context_size] for i in ix]
    x = [encode(char) for char  in x]
    x = torch.tensor(x)
    label = [val_text[i+1:i+context_size+1] for i in ix]
    label = [encode(char) for char in label ] 
    label = torch.tensor(label)
    return x , label
#A single self attention head 
class Head(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.size = size
        self.key = nn.Linear(embd_size,size, bias= False) 
        self.query = nn.Linear(embd_size,size, bias= False)
        self.value = nn.Linear(embd_size,size, bias= False)
        self.dropout = nn.Dropout(dropout) #we dropout and set some affinities to 0 so that the model doesnt overly really on any one token 
        

    def forward(self,x):
        k = self.key(x) #we perform linear transformation on x and get key vectors B T 4
        q = self.query(x)  #we perform linear transformation on x and get query vectors B T 4
        v = self.value(x) #we perform linear transformation on x and get value vectors B T 4
        self.weights = q @ k.transpose(-2,-1)*self.size ** -0.5 #we perform matrix multiplication on key and query matrice and scale it down by square root of head_Size do reduce variance for the softmax
        #B T 4 @ B 4 T -> B T T For every token this represents the affinity with other tokens in the same context
        tril = torch.tril(torch.ones(context_size,context_size))
        self.weights = self.weights.masked_fill(tril == 0, float('-inf')) #we perform triangular masking to disregard tokens that are positionally in front of the token we are analysing 
        self.weights = F.softmax(self.weights, dim= -1)
        self.weights = self.dropout(self.weights)
        output = self.weights @ v 
        return output
    

#We combine multiple heads into 1 multihead attention block
class MultiHeadAttention(nn.Module):
    def __init__(self,nb_heads,head_size):
        super().__init__()
        self.nb_heads = nb_heads
        self.head_size = head_size
        self.Heads = nn.ModuleList([Head(head_size) for i in range(nb_heads) ])
        self.proj = nn.Linear(embd_size,embd_size) #Without self.proj, the heads would remain separate and untransformed, which could limit the model's ability to integrate the insights gained from different attention heads.
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat ( [heads.forward(x) for heads in self.Heads], dim = 2  )
        out = self.proj(out)
        out = self.dropout(out)
        return out #We concatanate all the invidiual heads to form an embd_size mutli_head
        
    

#The outputs from the MHA are fedforward into the next block using an activation function and linear transformation
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size), nn.ReLU(),nn.Linear(4 * embd_size,embd_size), # multiplying by 4 allows to  increase the size of the skip connection pathway?
  nn.Dropout(dropout)      ) #We set some features to 0 so that the model doesnt overly really on any single feature of a token

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self,nb_heads,head_size):
        super().__init__()
        self.sa = MultiHeadAttention(nb_heads,head_size) 
        self.ffwd = FeedForward()
        self.ln_1 = nn.LayerNorm(embd_size)#We normalize the features of each token among themselves
        self.ln_2 = nn.LayerNorm(embd_size) 
    def forward(self,x):
        x = x + self.sa(self.ln_1(x)) #we provide a skip connection so that the gradient does not disappear for earlier layers
        return x +  self.ffwd(self.ln_2(x))
        
        

from torch.nn import functional as F

class TransformerModel(nn.Module):
    def __init__(self,embd_size): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,embd_size) #initializing the token embedding table for the identity of each token (embd_size,embd_size)
        self.positional_embedding_table = nn.Embedding(context_size ,embd_size) #initializing the positional embedding table for each position in the context 
        self.lm_head = nn.Linear(embd_size,vocab_size)
        self.blocks = nn.Sequential( *[Block(head_size,embd_size//head_size) for _ in range(n_layer)] ) 
        self.ln_f =  nn.LayerNorm(embd_size)
    def forward(self,idx,targets=None):
        token_embed_value = self.token_embedding_table(idx) 
        token_positional_value = self.positional_embedding_table(torch.arange(context_size,device=device)) 
        pos_and_embed_value = token_positional_value + token_embed_value #we sum the identity and the position of the token B , T, embd_size
        #4 8 32
        x = self.blocks(pos_and_embed_value) #we forward the values 
        logits = self.lm_head(x) #we transform the output into a class size array to get the logits
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits_flatten = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_flatten,targets)
        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            if context_size > len(idx[0]):
                new_token = torch.zeros((1,context_size - len(idx)), dtype = torch.long) #T 
                idx = torch.cat((idx,new_token),dim=1)
                logits,_  = self.forward(idx) 
            else:
                logits,_  = self.forward(idx[: , -context_size:]) # we get the logits for each Batch (B,T,C)
            logits = logits[:,-1,:] #convert logits to B,C (taking out the last column of every batch )
            proba = F.softmax(logits, dim=-1)  #convert the last dimension to probabilities
            new_token= torch.multinomial(proba,num_samples = 1) #generate new token based on proabilities (B,1)
            idx = torch.cat((idx,new_token),dim=1)
           
            
        return idx
            


m = TransformerModel(embd_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


#Training
#loss, optimizer and logits are connected through nn.Embedding
for iteration in range (10000):
    X, labels = get_batch() #We get a random batch 
    logits, loss = m.forward(X,targets=labels) #We calculaote the logits obtained from this batch 
    optimizer.zero_grad(set_to_none=True) #Not sure what this does
    loss.backward() #Computes the gradients of the loss
    optimizer.step() # Updates the weights of the model
print("Training loss")
print(loss.item())




#Text generation
idx = torch.zeros((1,1), dtype = torch.long) #Initalize with an empty character
print(decode(m.generate(idx,100)[0].tolist()))