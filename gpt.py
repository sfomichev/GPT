import os 
import sys
import torch
import numpy as np
import pickle
from torch.nn import functional as F

torch.manual_seed(1337)

block_size = 8
lr = 1e-3

max_iters =  5000
eval_iters = 200
eval_interval = 500
batch_size = 32

n_embd = 32
head_size = 16


dir_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv)>1:
    dir_path = sys.argv[1]

tr_ids = np.load(dir_path + '/tr_ids.npy')
val_ids = np.load(dir_path + '/val_ids.npy')

tr_ids = torch.tensor(tr_ids, dtype=torch.long)
val_ids = torch.tensor(val_ids, dtype=torch.long)

file = open(dir_path + '/meta.pkl','rb')
meta = pickle.load(file)

vocab_size = meta['vocab_size']
i_toch = meta['i_toch']
ch_toi = meta['ch_toi']

encode = lambda l: [ch_toi[ch] for ch in l]
decode = lambda l: [i_toch[i] for i in l]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split:str):
    data = tr_ids if split=='train' else val_ids
    idx = torch.randint(0,len(data)-block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])

    x, y = x.to(device), y.to(device)
    return x,y

#idx = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[i] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out


class AttentionHead(torch.nn.Module):
    def __init__(self, head_size, block_size = block_size):
        super().__init__()
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(2,1) * (T)**(-1/2)
        wei = wei.masked_fill(self.tril[:T, :T]==0, -float("inf"))
        wei = torch.softmax(wei, dim =-1)
        out = wei @ v

        return out
    
    
class BigramLanguageModule(torch.nn.Module):
    def __init__(self, vocab_size, n_embd=n_embd):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_emb = torch.nn.Embedding(vocab_size, n_embd)
        self.head = AttentionHead(head_size)

        self.lm_head = torch.nn.Linear(head_size,vocab_size)

    def forward(self, idx, targets=None):
        
        x = self.emb(idx)

        B,T = idx.shape
        pos = self.emb(torch.arange(0,T, device=device))
        x = x + pos

        x = self.head(x)

        logits = self.lm_head(x)

        #B,T,C = x.shape
        #logits = x.view(B*T,C)
                
        if targets is  None:
            loss= None
        else:            
            loss = F.cross_entropy(logits.view(-1,vocab_size), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:,-block_size:])
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            
            next_id = torch.multinomial(probs, num_samples=1)
            next_id = next_id.view(1,1)
            idx = torch.cat((idx, next_id), dim = 1)     
        return idx
    



model = BigramLanguageModule(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(),lr=lr)



for iter in range(max_iters):
    if iter % eval_interval == 0:
        out = estimate_loss()
        print(f"step {iter}: train loss = {out['train']}, val loss = {out['test']}")

    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    optimizer.step()


context = torch.zeros((1,block_size),dtype=torch.long, device=device)
print(''.join(decode(model.generate(idx=context ,max_new_tokens=1000)[0].tolist())))