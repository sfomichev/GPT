import os
import sys
import requests
import numpy as np
import pickle

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_dir = os.path.dirname(os.path.realpath(__file__))
# If there are args
if len(sys.argv)>1:
    data_dir = sys.argv[1]


# read and save file
if not os.path.isfile(data_dir+'/tinyshakespeare.txt'):
    r = requests.get(data_url, allow_redirects=True)
    open(data_dir+'/tinyshakespeare.txt', 'wb').write(r.content)

data = open(data_dir+'/tinyshakespeare.txt', 'r').read()

chars = sorted(list(set(data)))
ch_toi = {ch:i for i,ch in enumerate(chars)}
i_toch = {i:ch for i,ch in enumerate(chars)}

encode = lambda l: [ch_toi[ch] for ch in l]
decode = lambda l: [i_toch[i] for i in l]

len_of_dataset = len(data)
vocab_size = len(chars)

tr_data = data[:int(len_of_dataset * 0.8)]
val_data = data[int(len_of_dataset * 0.8):]

tr_ids = encode(tr_data)
val_ids = encode(val_data)

print(f"train has {len(tr_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")


tr_ids = np.array(tr_ids)
val_ids = np.array(val_ids)


np.save(data_dir+'/tr_ids' , tr_ids)
np.save(data_dir+'/val_ids' , val_ids)

meta = {
    'vocab_size': vocab_size,
    'i_toch': i_toch,
    'ch_toi': ch_toi,
}

with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)