import numpy
import torch
from torch import tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open("text.txt", "r") as f:
        text = f.read()
except FileNotFoundError:
    text = ""


text = text[:4000]

text = text.replace("\n", "")
text = text.replace("\t", " ")

vocab = set(list(text))

stoi = {c:i for i,c in enumerate(vocab)}
itos = {i:c for i,c in enumerate(vocab)}

lr = 1e-2
n_epochs = 100
vocab_size = len(vocab)
context_len = 64
embedding_dim = 64
feed_for_dim = 128
attention_dim = 16
num_heads = 4

embeddings = torch.randn(vocab_size,embedding_dim).to(device) *0.01

scale = torch.sqrt(1/torch.tensor(embedding_dim).to(device))
WQ = [torch.randn(embedding_dim,attention_dim).to(device)* scale for _ in range(num_heads)]
WK = [torch.randn(embedding_dim,attention_dim).to(device)* scale for _ in range(num_heads)]
WV = [torch.randn(embedding_dim,attention_dim).to(device)* scale for _ in range(num_heads)]

w_o = torch.randn(embedding_dim,embedding_dim).to(device) * scale

gamma_1 = torch.ones(1,embedding_dim).to(device)
gamma_2 = torch.ones(1,embedding_dim).to(device)

beta_1 = torch.zeros(1,embedding_dim).to(device)
beta_2 = torch.zeros(1,embedding_dim).to(device)

w1 = torch.randn(embedding_dim,feed_for_dim).to(device) * 0.01
w2 = torch.randn(feed_for_dim,embedding_dim).to(device) * 0.01

b1 = torch.zeros(1,feed_for_dim).to(device)
b2 = torch.zeros(1,embedding_dim).to(device)

w_p = torch.randn(embedding_dim,vocab_size).to(device) * 0.01
b_p = torch.zeros(1,vocab_size).to(device)

mask = torch.where(torch.triu(torch.ones(context_len, context_len).to(device), diagonal=1).bool(),float('-inf'),0.0)

def positional_encoding(seq_len):
    PE = torch.zeros(seq_len, embedding_dim, device=device)

    for pos in range(seq_len):
        for k in range(0, embedding_dim, 2):
            div = 10000 ** (k / embedding_dim)
            val = torch.tensor(pos / div, device=device)

            PE[pos, k] = torch.sin(val)
            PE[pos, k + 1] = torch.cos(val)

    return PE

def softmax(input):
    e = torch.exp(input - torch.max(input, dim=-1, keepdim=True).values)
    return e / torch.sum(e, dim=-1, keepdim=True)

def LayerNorm(input,gamma,beta):
    mean = torch.mean(input, dim=-1, keepdim=True)
    var = torch.var(input, dim=-1, keepdim=True,unbiased=False)

    in_norm = (input-mean)/torch.sqrt(var+1e-8)

    normed = gamma * in_norm + beta

    return normed,in_norm,var,mean

for epoch in range(n_epochs):
    input = text[:context_len]
    encoded_input = [stoi[c] for c in input]

    X = embeddings[encoded_input] + positional_encoding(context_len)
    att_in_X,in_norm1,var_1,mean1 = LayerNorm(X,gamma_1,beta_1)

    Q_list,K_list,V_list,scores_list,weights_list,outs_list = [],[],[],[],[],[]

    att_scale = torch.sqrt(torch.tensor(attention_dim).to(device))

    for w_q,w_k,w_v in zip(WQ,WK,WV):
        Q = att_in_X @ w_q
        K = att_in_X @ w_k
        V = att_in_X @ w_v

        scores = Q @ K.T / att_scale
        weights = softmax(scores+mask)
        out = weights @ V

        Q_list.append(Q)
        K_list.append(K)
        V_list.append(V)
        scores_list.append(scores)
        weights_list.append(weights)
        outs_list.append(out)

    concated = torch.cat(outs_list, dim=-1)

    att_out = concated @ w_o
    att_out = att_out + X

    feed_for_in_att_out,in_norm2,var_2,mean_2 = LayerNorm(att_out,gamma_2,beta_2)

    z1 = feed_for_in_att_out @ w1 + b1
    a1 = torch.maximum(z1,torch.zeros_like(z1))
    z2 = a1 @ w2 + b2

    z2 = att_out + z2

    projected = z2 @ w_p + b_p

    probs = softmax(projected)

    targets = text[1:context_len+1]
    encoded_targets = [stoi[c] for c in targets]

    loss = []

    for prob,target in zip(probs,encoded_targets):
        loss.append(-torch.log(prob[target] + 1e-8))

    loss = torch.stack(loss).mean()

    print(f"loss: {loss}")

    d_logits = probs.clone()

    for d_logit,target in zip(d_logits, encoded_targets):
        d_logit[target] -= 1

    d_logits /= len(encoded_targets)

    # Projection
    d_projected = d_logits

    d_wp = z2.T @ d_projected
    d_bp = d_projected.sum(dim=0,keepdim=True)
    ##

    d_z2 = d_projected @ w_p.T

    # Feed Forward
    d_w2 = a1.T @ d_z2
    d_b2 = d_z2.sum(dim=0,keepdim=True)

    d_a1 = d_z2 @ w2.T

    d_z1 = d_a1 * (z1 > 0).float()

    d_w1 = feed_for_in_att_out.T @ d_z1
    d_b1 = d_z1.sum(dim=0,keepdim=True)
    ##
    d_att_out = d_z1 @ w1.T

    d_att_out += d_z2

    d_normed2 = d_att_out

    # Layer Norm
    d_gamma_2 = torch.sum(d_normed2 * in_norm2, dim=0, keepdim=True)
    d_beta_2 = d_normed2.sum(dim=0, keepdim=True)

    d_in_norm2 = gamma_2 * d_normed2

    std_2 = torch.sqrt(var_2 + 1e-8)

    d_input = (1 / std_2) * (d_in_norm2- d_in_norm2.mean(dim=-1, keepdim=True)- in_norm2 * (d_in_norm2 * in_norm2).mean(dim=-1, keepdim=True))

    ##
    d_att_aft = d_input

    # Attention
    d_wo = concated.T @ d_att_aft
    d_concated = d_att_aft @ w_o.T

    d_outs_list = torch.split(d_concated, attention_dim, dim=-1)

    d_X = torch.zeros(context_len,embedding_dim).to(device)

    d_WQ,d_WK,d_WV = [],[],[]

    for d_out,Q,K,V,weights,scores,w_q,w_k,w_v in zip(d_outs_list,Q_list,K_list,V_list,weights_list,scores_list,WQ,WK,WV):
        d_weights = d_out @ V.T
        d_V = weights.T @ d_out

        d_scores = weights * (d_weights - torch.sum(d_weights * weights, dim=-1, keepdim=True))

        d_Q = (d_scores @ K) / att_scale
        d_K = (d_scores.T @ Q) / att_scale

        d_wv = att_in_X.T @ d_V
        d_wk = att_in_X.T @ d_K
        d_wq = att_in_X.T @ d_Q

        d_X += d_V @ w_v.T
        d_X += d_K @ w_k.T
        d_X += d_Q @ w_q.T

        d_WQ.append(d_wq)
        d_WK.append(d_wk)
        d_WV.append(d_wv)

    ##
    d_normed1 = d_X

    # Layer Norm 1

    d_gamma_1 = torch.sum(d_normed1 * in_norm1, dim=0, keepdim=True)
    d_beta_1 = d_normed1.sum(dim=0, keepdim=True)

    d_in_norm1 = gamma_1 * d_normed1

    std_1 = torch.sqrt(var_1 + 1e-8)

    d_input = (1 / std_1) * (d_in_norm1- d_in_norm1.mean(dim=-1, keepdim=True)- in_norm1 * (d_in_norm1 * in_norm1).mean(dim=-1, keepdim=True))

    ##
    d_embed_pe = d_input + d_att_aft

# Embeddings
    d_embeddings = torch.zeros_like(embeddings)

    indices = torch.tensor(encoded_input, device=device)

    d_embeddings.index_add_(0, indices, d_embed_pe)

    ##

    # Step

    embeddings -= lr * d_embeddings

    w_p -= lr * d_wp
    b_p -= lr * d_bp

    w2 -= lr * d_w2
    b2 -= lr * d_b2

    w1 -= lr * d_w1
    b1 -= lr * d_b1

    w_o -= lr * d_wo

    gamma_2 -= lr * d_gamma_2
    beta_2 -= lr * d_beta_2

    gamma_1 -= lr * d_gamma_1
    beta_1 -= lr * d_beta_1

    for i in range(num_heads):
        WQ[i] -= lr * d_WQ[i]
        WK[i] -= lr * d_WK[i]
        WV[i] -= lr * d_WV[i]

##

