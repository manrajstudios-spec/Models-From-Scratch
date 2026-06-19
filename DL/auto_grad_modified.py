import torch

class Node:
    def __init__(self, val):
        self.val = val
        self.grad = torch.zeros_like(val)

class AutoGrad:
    def __init__(self):
        self.back_list = []
        self.nodes = set()

    def track(self, *nodes):
        for n in nodes:
            self.nodes.add(n)

    def zero_grad(self):
        for node in self.nodes:
            node.grad = 0.0

    def clear(self):
        self.back_list.clear()
        self.nodes.clear()

    def backward(self):
        for back in self.back_list[::-1]:
            back()

    def step(self, lr):
        for node in self.nodes:
            node.val -= lr * node.grad

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
