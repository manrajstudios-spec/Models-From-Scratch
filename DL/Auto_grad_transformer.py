import re
import torch
import auto_grad_modified

class Node:
    def __init__(self, x,b=False ,y=0):
        self.val = x
        self.grad = y
        self.acc = b

with open("text.txt", "r") as f:
    text = f.read()

text = re.sub(r'\n+', '\n', text)
text = re.sub(r' +', ' ', text)
text = text.strip()
text = text[:10000]

split = int(len(text) * 0.7)
train_text = text[:split]
val_text = text[split:]

chars = sorted(list(set(text)))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
embedding_dim = 64
attention_dim = 32
feed_for_dim = 128
context_len = 64
n_heads = 2

embeddings = torch.randn(vocab_size, embedding_dim)*0.01

def positional_encoding(seq_len):
    pe = torch.zeros(seq_len, embedding_dim)
    pos = torch.arange(seq_len).unsqueeze(1)

    i = torch.arange(0, embedding_dim, 2)
    div = torch.pow(10000, 2 * i / embedding_dim)

    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)

    return pe

pe = positional_encoding(context_len)

auto_grad = auto_grad_modified.AutoGrad()

scale = 1.0 / torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float))

w_q_A = [Node(torch.randn(embedding_dim,attention_dim) * scale) for _ in range(n_heads)]
w_k_A = [Node(torch.randn(embedding_dim,attention_dim) * scale) for _ in range(n_heads)]
w_v_A = [Node(torch.randn(embedding_dim,attention_dim) * scale) for _ in range(n_heads)]
w_o = Node(torch.randn(n_heads * attention_dim, embedding_dim))

w1 = Node(torch.randn(embedding_dim,feed_for_dim) * 0.01)
b1 = Node(torch.zeros(1,feed_for_dim))
w2 = Node(torch.randn(feed_for_dim,embedding_dim) * 0.01)
b2 = Node(torch.zeros(1,embedding_dim))

w_p = Node(torch.randn(embedding_dim, vocab_size) * 0.01)
b_p = Node(torch.zeros(1, vocab_size))

mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))

for epoch in range(100):
    for s in range(0,len(train_text) - context_len - 1 ,context_len):
        chunk = train_text[s:s+context_len]
        encoded_chunk = [stoi[c] for c in chunk]
        encoded_targets = torch.tensor([stoi[c] for c in train_text[s + 1:s + context_len + 1]])

        X = Node(embeddings[encoded_chunk] + pe)

        Q_list,K_list,V_list,scores_list,weights_list,out_list = [],[],[],[],[],[]

        for w_q,w_k,w_v in zip(w_q_A,w_k_A,w_v_A):
            Q = auto_grad.mat_mul(X, w_q)
            K = auto_grad.mat_mul(X, w_k)
            V = auto_grad.mat_mul(X, w_v)

            scores = auto_grad.mat_mul(Q,K,b_transpose=True)
            scores = auto_grad.scale(scores,1/torch.sqrt(torch.tensor(attention_dim, dtype=torch.float)))
            scores.val += mask
            weights = auto_grad.softmax(scores)
            outs = auto_grad.mat_mul(weights, V)

            Q_list.append(Q)
            K_list.append(K)
            V_list.append(V)
            scores_list.append(scores)
            weights_list.append(weights)
            out_list.append(outs)

        concat = auto_grad.concat(out_list)
        att_out = auto_grad.mat_mul(concat, w_o)

        z1 = auto_grad.mat_add(auto_grad.mat_mul(att_out,w1),b1,True)
        a1 = auto_grad.relu(z1)
        z2 = auto_grad.mat_add(auto_grad.mat_mul(a1,w2),b2,True)

        projected_x = auto_grad.mat_add(auto_grad.mat_mul(z2,w_p),b_p,True)

        loss, probs = auto_grad.softmax_cross_entropy(projected_x, encoded_targets)

        auto_grad.backward()
        auto_grad.step(0.01)
        auto_grad.zero_grad()
        auto_grad.clear()
