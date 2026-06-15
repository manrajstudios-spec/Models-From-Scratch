import numpy as np
import re
with open("text.txt", "r") as f:
    text = f.read()

text = re.sub(r'\n+', '\n', text)
text = re.sub(r' +', ' ', text)
text = text.strip()
text = text[:10000]

chars = sorted(list(set(text)))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

temp = 0.8
lr = 0.01
vocab_size = len(chars)
dim_model = 128
context_len = 64
n_heads = 4
attention_dim = 32
feed_for_dim = 512

split = int(len(text) * 0.7)
train_text = text[:split]
val_text = text[split:]

embeddings = np.random.randn(vocab_size, dim_model) * 0.01

encoded = [stoi[c] for c in train_text]
targets_char = train_text[1:context_len+1]
encoded_targets = [stoi[c] for c in targets_char]

scale = 1.0 / np.sqrt(dim_model)

w_qs = [np.random.randn(dim_model, attention_dim) * scale for _ in range(n_heads)]
w_ks = [np.random.randn(dim_model, attention_dim) * scale for _ in range(n_heads)]
w_vs = [np.random.randn(dim_model, attention_dim) * scale for _ in range(n_heads)]
w_o  = np.random.randn(n_heads * attention_dim, dim_model) * scale

w1 = np.random.randn(dim_model, feed_for_dim) * 0.01
w2 = np.random.randn(feed_for_dim, dim_model) * 0.01

b1 = np.zeros((1, feed_for_dim))
b2 = np.zeros((1, dim_model))

w_p = np.random.randn(dim_model, vocab_size) * 0.01
b_p = np.zeros((1, vocab_size))

def softmax(x):
    e = np.exp(x - np.max(x,axis=-1,keepdims=True))
    return e / e.sum(axis=1,keepdims=True)

def position_encoding(seq_len):
    pe = np.zeros((seq_len, dim_model))

    for pos in range(seq_len):
        for i in range(0,dim_model,2):
            pe[pos, i] = np.sin(pos / (10000 ** (2 * i / dim_model)))
            pe[pos, i+1] = np.cos(pos / (10000 ** (2 * i / dim_model)))

    return pe

def multi_attention(X, mask):
    head_outs, weights_list, Q_list, K_list, V_list = [], [], [], [], []

    for w_q, w_k, w_v in zip(w_qs, w_ks, w_vs):
        Q = X @ w_q
        K = X @ w_k
        V = X @ w_v
        scores = (Q @ K.T) / np.sqrt(attention_dim)
        weights = softmax(scores + mask)
        head_outs.append(weights @ V)
        weights_list.append(weights)
        Q_list.append(Q)
        K_list.append(K)
        V_list.append(V)

    concat = np.concatenate(head_outs, axis=1)
    out = concat @ w_o
    return out, concat, weights_list, Q_list, K_list, V_list

def feed_forward(X):
    global w1, b1, w2, b2
    z1 = X @ w1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2

    return z2, a1, z1

def project_to_vocab(X):
    global w_p, b_p
    p = X @ w_p + b_p
    return p

def cross_entropy(logits, targets):
    probs = softmax(logits)
    correct_probs = probs[np.arange(len(targets)), targets]
    loss = -np.mean(np.log(correct_probs + 1e-8))
    return loss, probs

def cross_entropy_backward(probs, targets):
    seq_len = len(targets)
    d_logits = probs.copy()
    d_logits[np.arange(seq_len), targets] -= 1
    d_logits /= seq_len
    return d_logits

def softmax_backward(d_out, weights):
    s = (d_out * weights).sum(axis=-1, keepdims=True)
    d_scores = weights * (d_out - s)

    return d_scores

pe = position_encoding(context_len)

mask = np.triu(np.ones((context_len, context_len)), k=1)
mask[mask == 1] = -np.inf

for epoch in range(500):
    for start in range(0, len(train_text) - context_len - 1, context_len):
        encoded = [stoi[c] for c in train_text[start:start + context_len]]
        encoded_targets = [stoi[c] for c in train_text[start + 1:start + context_len + 1]]
        X = embeddings[encoded] + pe

        att_out, concat, weights_list, Q_list, K_list, V_list = multi_attention(X, mask)
        att_out = att_out + X

        z2,a1,z1 = feed_forward(att_out)
        z2 = z2 + att_out

        logits = project_to_vocab(z2)

        loss,probs = cross_entropy(logits, encoded_targets)

        d_logits = cross_entropy_backward(probs, encoded_targets)

        dw_p = z2.T @ d_logits
        db_p = d_logits.sum(axis=0, keepdims=True)

        d_z2 = d_logits @ w_p.T

        d_w2 = a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = d_z2 @ w2.T
        d_z1 = d_a1 * (z1 > 0)

        d_w1 = att_out.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        d_att_out = d_z1 @ w1.T
        d_att_out += d_z2

        d_wo = concat.T @ d_att_out
        d_concat = d_att_out @ w_o.T

        d_X = np.zeros_like(X)
        d_wqs, d_wks, d_wvs = [], [], []

        for i, (w_q, w_k, w_v, weights, Q, K, V) in enumerate(zip(w_qs, w_ks, w_vs, weights_list, Q_list, K_list, V_list)):
            d_head = d_concat[:, i * attention_dim:(i + 1) * attention_dim]

            d_V = weights.T @ d_head
            d_scores = softmax_backward(d_head @ V.T, weights) / np.sqrt(attention_dim)
            d_Q = d_scores @ K
            d_K = d_scores.T @ Q

            d_wqs.append(X.T @ d_Q)
            d_wks.append(X.T @ d_K)
            d_wvs.append(X.T @ d_V)

            d_X += d_Q @ w_q.T + d_K @ w_k.T + d_V @ w_v.T

        w_o -= lr * d_wo
        d_X += d_att_out
        w1 -= lr * d_w1
        b1 -= lr * d_b1
        w2 -= lr * d_w2
        b2 -= lr * d_b2

        for i in range(n_heads):
            w_qs[i] -= lr * d_wqs[i]
            w_ks[i] -= lr * d_wks[i]
            w_vs[i] -= lr * d_wvs[i]

        w_p -= lr * dw_p
        b_p -= lr * db_p

        np.add.at(embeddings, encoded, -lr * d_X)

    if epoch % 100 == 0:
        val_start = np.random.randint(0, len(val_text) - context_len - 1)
        val_chunk = val_text[val_start:val_start + context_len]
        val_encoded = [stoi[c] for c in val_chunk]
        val_targets = [stoi[c] for c in val_text[val_start + 1:val_start + context_len + 1]]

        x_val = embeddings[val_encoded] + pe
        att_val, concat_val, wl, Ql, Kl, Vl = multi_attention(x_val, mask)
        att_val = att_val + x_val
        ff_val, _, _ = feed_forward(att_val)
        ff_val = ff_val + att_val
        logits_val = project_to_vocab(ff_val)
        val_loss, _ = cross_entropy(logits_val, val_targets)

        print(f"epoch:{epoch} | train:{loss:.4f} | val:{val_loss:.4f}")

def predict_one(input):
    encoded_in = [stoi[c] for c in input]
    X = embeddings[encoded_in] + pe
    att_out, concat, weights_list, Q_list, K_list, V_list = multi_attention(X, mask)
    att_out = att_out + X

    z2, a1, z1 = feed_forward(att_out)
    z2 = z2 + att_out

    logits = project_to_vocab(z2)

    probs = softmax(logits[-1:]/temp)
    next_tok = np.random.choice(vocab_size,p=probs[0])

    return itos[next_tok]


def predict(max_len):
    buf = list(text[:context_len])

    for i in range(max_len):
        p = predict_one("".join(buf))
        buf.append(p)
        buf = buf[-context_len:]

    print("".join(buf))

predict(100)