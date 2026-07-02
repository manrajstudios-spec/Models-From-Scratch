import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node:
    def __init__(self, val):
        self.val = val
        self.grad = torch.zeros_like(val).to(device=device)
        self.m = torch.zeros_like(val).to(device=device)
        self.v = torch.zeros_like(val).to(device=device)

class AutoGrad:
    def __init__(self):
        self.back_list = []
        self.nodes = set()
        self.t = 0

    def track(self, *nodes):
        for n in nodes:
            self.nodes.add(n)

    def zero_grad(self):
        for node in self.nodes:
            node.grad = torch.zeros_like(node.val).to(device=device)

    def clear(self):
        self.back_list.clear()
        self.nodes.clear()

    def backward(self):
        for back in self.back_list[::-1]:
            back()

    def step(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t+=1
        for node in self.nodes:
            node.m = beta1 * node.m + (1 - beta1) * node.grad
            node.v = beta2 * node.v + (1 - beta2) * node.grad ** 2

            m_hat = node.m / (1 - beta1 ** self.t)
            v_hat = node.v / (1 - beta2 ** self.t)

            node.val -= lr * m_hat / (torch.sqrt(v_hat) + eps)

    def multiply(self, a, b):
        out = Node(a.val * b.val)

        def backward():
            a.grad += out.grad * b.val
            b.grad += a.val * out.grad

        self.back_list.append(backward)
        self.track(a, b, out)

        return out

    def add(self, a, b):
        out = Node(a.val + b.val)

        def backward():
            a.grad += out.grad
            b.grad += out.grad

        self.back_list.append(backward)
        self.track(a, b, out)

        return out

    def mat_mul(self, a, b, a_transpose=False, b_transpose=False):
        val_a = a.val.transpose(-2, -1) if a_transpose else a.val
        val_b = b.val.transpose(-2, -1) if b_transpose else b.val

        out = Node(val_a @ val_b)

        def backward():
            grad_a = out.grad @ val_b.transpose(-2, -1)
            grad_b = val_a.transpose(-2, -1) @ out.grad

            if a_transpose:
                grad_a = grad_a.transpose(-2, -1)

            if b_transpose:
                grad_b = grad_b.transpose(-2, -1)

            a.grad += grad_a
            b.grad += grad_b

        self.back_list.append(backward)
        self.track(a, b, out)

        return out

    def mat_add(self, a, b, bias=False):
        out = Node(a.val + b.val)

        def backward():
            a.grad += out.grad

            if bias:
                b.grad += out.grad.sum(dim=0, keepdim=True)
            else:
                b.grad += out.grad

        self.back_list.append(backward)
        self.track(a, b, out)

        return out

    def divide(self,a,b):
        out = Node(a.val / b.val)

        def back_ward():
            a.grad += out.grad / b.val
            b.grad -= out.grad * a.val / (b.val ** 2)

        out.back_ward = back_ward

        self.back_list.append(back_ward)
        self.track(a, b, out)

        return out

    def sqrt(self,X):
        out = Node(torch.sqrt(X.val))

        def back_ward():
            X.grad += out.grad * (0.5 * (1/out.val))

        out.back_ward = back_ward
        self.back_list.append(back_ward)
        self.track(X,out)

        return out

    def scale(self, a, sc):
        out = Node(a.val * sc)

        def backward():
            a.grad += out.grad * sc

        self.back_list.append(backward)
        self.track(a, out)

        return out

    def relu(self, a):
        out = Node(torch.clamp(a.val, min=0))

        def backward():
            a.grad += out.grad * (a.val > 0)

        self.back_list.append(backward)
        self.track(a, out)

        return out

    def tanh(self, a):
        t = torch.tanh(a.val)
        out = Node(t)

        def backward():
            a.grad += out.grad * (1 - t ** 2)

        self.back_list.append(backward)
        self.track(a, out)

        return out

    def sigmoid(self, a):
        s = 1 / (1 + torch.exp(-a.val))
        out = Node(s)

        def backward():
            a.grad += out.grad * s * (1 - s)

        self.back_list.append(backward)
        self.track(a, out)

        return out

    def softmax(self, a):
        e = torch.exp(a.val - torch.max(a.val, dim=-1, keepdim=True).values)
        probs = e / e.sum(dim=-1, keepdim=True)
        out = Node(probs)

        def backward():
            s = (out.grad * probs).sum(dim=-1, keepdim=True)
            a.grad += probs * (out.grad - s)

        self.back_list.append(backward)
        self.track(a, out)

        return out

    def softmax_cross_entropy(self, logits, targets):
        e = torch.exp(logits.val - torch.max(logits.val, dim=-1, keepdim=True).values)
        probs = e / e.sum(dim=-1, keepdim=True)
        correct = probs[torch.arange(len(targets)), targets]

        loss_val = -torch.mean(torch.log(correct + 1e-8))
        out = Node(loss_val)
        out.grad = 1.0

        def backward():
            d = probs.clone()
            d[torch.arange(len(targets)), targets] -= 1
            d /= len(targets)
            logits.grad += out.grad * d

        self.back_list.append(backward)
        self.track(logits, out)

        return out, probs

    def mse(self, pred, target):
        out = Node((pred.val - target) ** 2)
        out.grad = 1.0

        def backward():
            pred.grad += out.grad * 2 * (pred.val - target)

        self.back_list.append(backward)
        self.track(pred, out)

        return out

    def concat(self, nodes, dim=-1):
        vals = [n.val for n in nodes]
        out = Node(torch.cat(vals, dim=dim))
        sizes = [n.val.shape[dim] for n in nodes]

        def backward():
            grads = torch.split(out.grad, sizes, dim=dim)
            for n, g in zip(nodes, grads):
                n.grad += g

        self.back_list.append(backward)
        self.track(*nodes, out)
        return out

    def layer_norm(self, X, g, b):
        mean = X.val.mean(axis=-1, keepdims=True)
        std = X.val.std(axis=-1, keepdims=True) + 1e-5
        x_norm = (X.val - mean) / std

        out = Node(g.val * x_norm + b.val)

        def backward():
            N = X.val.shape[-1]
            dout = out.grad * g.val

            dx_norm = dout
            dstd = (dx_norm * (X.val - mean) * -0.5 * std ** -3).sum(axis=-1, keepdims=True)
            dmean = (-dx_norm / std).sum(dim=-1, keepdim=True)

            X.grad += dx_norm / std + dstd * 2 * (X.val - mean) / N + dmean / N
            g.grad += (out.grad * x_norm).sum(dim=0)
            b.grad += out.grad.sum(dim=0)

        out._backward = backward
        self.back_list.append(backward)
        self.track(X,out,g,b)

        return out

    def embed(self, embeddings, indices):
        out = Node(embeddings.val[indices])

        def backward():
            for i, idx in enumerate(indices):
                embeddings.grad[idx] += out.grad[i]

        self.back_list.append(backward)
        self.track(embeddings, out)

        return out