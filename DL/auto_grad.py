import numpy as np

class Node:
    def __init__(self, x):
        self.val = x
        self.grad = np.zeros_like(x)
        self.back_ward = lambda: None

back_list = []

def mat_mul(a, b,a_transpose,b_transpose):
    val_a = a.val.T if a_transpose else a.val
    val_b = b.val.T if b_transpose else b.val

    out = Node(val_a@val_b)

    def backward():
        grad_a = out.grad @ val_b.T
        grad_b = val_a.T @ out.grad

        grad_a = grad_a.T if a_transpose else grad_a
        grad_b = grad_b.T if b_transpose else grad_b

        a.grad += grad_a
        b.grad += grad_b

    return out

def mat_add(a,b,bias=False):
    out = Node(a.val + b.val)

    def backward():
        a.grad += out.grad

        if bias:
            b.grad += out.grad.sum(axis=1,keepdims=True)
        else:
            b.grad += out.grad

    back_list.append(backward)
    out.back_ward = backward

    return out

def scale(X,sc):
    out = Node(X.val * sc)

    def backward():
        X.grad += out.grad * sc

    back_list.append(backward)
    out.back_ward = backward

    return out

def relu(X):
    out = Node(np.maximum(0,X.val))

    def backward():
        X.grad += out.grad * (1 if X.val >= 0 else 0)

    back_list.append(backward)
    out.back_ward = backward

    return out

def sigmoid(X):
    out = Node(1/1+np.exp(-X.val))

    def backward():
        X.grad += out.grad * (out.val * (1 - out.val))

    back_list.append(backward)
    out.back_ward = backward

    return out

def tanh(X):
    out = Node(np.tanh(X.val))

    def backward():
        X.grad += out.grad * (1-out.val ** 2)

    back_list.append(backward)
    out.back_ward = backward

    return out

def softmax(X):
    s = np.exp(X.val - X.val.max(keepdims=True,axis=-1))
    out = Node(s/s.sum(axis=-1,keepdims=True))

    def backward():
        s = np.sum(out.grad * out.val, axis=-1, keepdims=True)
        local_grad = out.val * (out.grad - s)

        X.grad += local_grad

    back_list.append(backward)
    out.back_ward = backward

    return out

def softmax_cross_entropy(X, targets):
    exp_x = np.exp(X.val - np.max(X.val, axis=-1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    loss_val = -np.log(probs[np.arange(X.val.shape[0]), targets]).mean()

    out = Node(loss_val)

    def backward():
        grad = probs.copy()

        grad[np.arange(X.val.shape[0]), targets] -= 1

        grad /= X.val.shape[0]

        X.grad += out.grad * grad

    back_list.append(backward)
    out.back_ward = backward

    return out, probs
