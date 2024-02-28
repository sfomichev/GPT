import os 
import sys
import torch
import numpy as np
import pickle
from torch.nn import functional as F

torch.manual_seed(1337)

batch_size = 32
block_size = 256
max_iters =  6000
eval_interval = 600
eval_iters = 200
lr = 3e-4
n_embd = 384

n_head = 6
n_layer = 6
dropout = 0.2


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

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * (C)**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T]==0, -float("inf"))
        wei = F.softmax(wei, dim =-1)
        wei = self.dropout(wei)
        out = wei @ v

        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_of_heads, head_size):
        super().__init__()

        self.mylti_head = torch.nn.ModuleList([AttentionHead(head_size=head_size) for _ in range(num_of_heads)])
        self.proj = torch.nn.Linear(n_embd,n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        x = [h(x) for h in self.mylti_head]
        x = torch.cat(x, dim = -1)
        x = self.dropout(self.proj(x))

        return x
    
class FeedForward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd,n_embd),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
    
class Block( torch.nn.Module):
    def __init__(self, n_embd, num_of_heads):
        super().__init__()        
        head_size = n_embd // num_of_heads
        self.sa_heads =  MultiHeadAttention(num_of_heads, head_size)
        self.FF = FeedForward(n_embd)

        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa_heads( self.ln1(x) )
        x = x + self.FF( self.ln2(x) )
        return x



class GPTLanguageModel(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, n_embd)
        self.pos_emb = torch.nn.Embedding(vocab_size, n_embd)
        # self.blocks = torch.nn.Sequential(
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     torch.nn.LayerNorm(n_embd),
        # )
        self.blocks = torch.nn.Sequential(*[Block(n_embd, num_of_heads=n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        
        B,T = idx.shape

        x = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(0,T, device=device))
        x = x + pos        

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)
                
        if targets is  None:
            loss= None
        else:      
            logits = logits.view(-1,vocab_size)      
            targets = targets.view(-1)     
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim = 1)     
        return idx
    



model = GPTLanguageModel()
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