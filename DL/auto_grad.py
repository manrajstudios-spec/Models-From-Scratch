class Node:
    def __init__(self, x,b=False ,y=0):
        self.val = x
        self.grad = y
        self.acc = b

back_list = []

def mul(a,b):
    mul = a.val * b.val
    a.grad = b.val
    b.grad = a.val
    back_list.append([a,b])

    return Node(mul,True)

def add(a,b):
    sume = a.val + b.val
    a.grad = 1
    b.grad = 1

    back_list.append([a,b])
    return Node(sume,True)

def relu(a):
    r = max(0,a.val)
    a.grad = 1 if a.val >= 0 else 0
    back_list.append([a])
    return Node(r,True)

def loss(pred,y):
    l = (pred.val - y) ** 2
    pred.grad = 2 * (pred.val - y)
    back_list.append([pred])
    return l

x = Node(2,True)
w1 = Node(3,False)
w2 = Node(4,False)
b1 = Node(2,False)
b2 = Node(7,False)
y = 10

z1 = add(mul(x,w1),b1)
a1 = relu(z1)
z2 = add(mul(a1,w2),b2)
l = loss(z2,y)

grad = 1
saved = []

for b in back_list[::-1]:
    temp = grad
    for x in b:
        if x.acc:
            grad = grad * x.grad
            print(f"grad: {grad}, temp: {temp}, X.grad: {x.grad}")
        else:
            saved.append({"value": x.val, "grad": x.grad * temp})
            print(f"x.grad: {x.grad * temp}")

print(saved)

