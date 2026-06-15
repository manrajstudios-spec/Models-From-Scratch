import numpy as np

text = "hello world hello world hello world hello world hello world Yo Bzro brop yo yes hi"

chars = list(set(text))

mapped_chars = {c:i for i,c in enumerate(chars)}

char_mapped = {i:c for c,i in mapped_chars.items()}

lr = 0.1
hidden_size = 1240
seq_len = len(text) - 1

w_x = np.random.randn(hidden_size,len(chars)) * 0.01

w_h = np.random.randn(hidden_size,hidden_size) * 0.01
b_h = np.zeros((hidden_size,1))

w_y = np.random.randn(len(chars),hidden_size) * 0.01
b_y = np.zeros((len(chars),1))

def encode(char):
    return mapped_chars.get(char,-1)

def decode(char):
    return char_mapped.get(char,"?")

def one_hot(idx):
    onehot = np.zeros((len(chars),1))
    onehot[idx] = 1
    return onehot

def soft_max(y):
    e = np.exp(y - y.max())
    return e / e.sum()

def cross_entropy(probs,val):
    return -np.log(probs[val,0] + 1e-12)

for epoch in range(1000):
    total_loss = 0
    h = np.zeros((hidden_size, 1))

    x_p = []
    h_p = []
    a_p = []
    probs_p = []
    correct_p = []

    for i,c in enumerate(text[:-1]):
        x = one_hot(encode(c))
        z = np.dot(w_x,x) + np.dot(w_h,h) + b_h
        a = np.tanh(z)

        y = np.dot(w_y,a) + b_y
        probs = soft_max(y)

        loss = cross_entropy(probs,encode(text[i+1]))
        total_loss += loss

        x_p.append(x)
        h_p.append(h.copy())
        a_p.append(a)
        probs_p.append(probs)
        correct_p.append(one_hot(encode(text[i+1])))

        h = a

    dw_x = np.zeros_like(w_x)

    dw_h = np.zeros_like(w_h)
    d_h = np.zeros_like(h)
    db_h = np.zeros_like(b_h)

    dw_y = np.zeros_like(w_y)
    db_y = np.zeros_like(b_y)

    dh_n = np.zeros((hidden_size,1))

    for i in reversed(range(len(text[:-1]))):
        dy = probs_p[i] - correct_p[i]

        dw_y += np.dot(dy,a_p[i].T)
        db_y += dy

        dh = np.dot(w_y.T,dy) + dh_n
        da = dh * (1-a_p[i]**2)

        dw_x += np.dot(da,x_p[i].T)
        dw_h += np.dot(da,h_p[i].T)
        db_h += da

        dh_n = np.dot(w_h.T,da)

    # for grad in [dw_x, dw_h, dw_y, db_h, db_y]:
    #     np.clip(grad, -1, 1, out=grad)

    dw_x /= seq_len
    dw_h /= seq_len
    dw_y /= seq_len
    db_h /= seq_len
    db_y /= seq_len

    w_x -= lr * dw_x
    w_h -= lr * dw_h
    w_y -= lr * dw_y
    b_h -= lr * db_h
    b_y -= lr * db_y

    print(f"epoch {epoch}, total loss: {total_loss}")
h = np.zeros((hidden_size, 1))

for i,c in enumerate(text[:-1]):
    x = one_hot(encode(c))
    z = np.dot(w_x,x) + np.dot(w_h,h) + b_h
    a = np.tanh(z)

    y = np.dot(w_y,a) + b_y
    probs = soft_max(y)

    print(f"prediction: {decode(probs.argmax())}; actual: {text[i+1]}")