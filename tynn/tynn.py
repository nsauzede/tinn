from math import exp
from dataclasses import dataclass, field

class Tynn:
    # All the weights.
    w:list[float]=field(default_factory=list)
    # Offset of hidden weights into all weights
    xofs:int
    # Biases.
    b:list[float]=field(default_factory=list)
    # Hidden layer.
    h:list[float]=field(default_factory=list)
    # Output layer.
    o:list[float]=field(default_factory=list)
    # Number of biases - always two - Tinn only supports a single hidden layer.
    nb:int
    # Number of weights.
    nw:int
    # Number of inputs.
    nips:int
    # Number of hidden neurons.
    nhid:int
    # Number of outputs.
    nops:int

def err(a:float, b:float) -> float:
    """Computes error."""
    return float(0.5) * (a - b) * (a - b)

def pderr(a:float, b:float) -> float:
    """Returns partial derivative of error function."""
    return a - b

def toterr(tgt:list[float], o:list[float], size:int) -> float:
    """Computes total error of target to output."""
    sum = float(0)
    for i in range(size):
        sum += err(tgt[i], o[i])
    return sum

def act(a:float) -> float:
    """Activation function."""
    return float(1) / (float(1) + exp(-a))

def pdact(a:float) -> float:
    """Returns partial derivative of activation function."""
    return a * (float(1) - a)

def bprop(t:Tynn, inp:list[float], tgt:list[float], rate:float):
    """Performs back propagation."""
    for i in range(t.nhid):
        sum = float(0)
        # Calculate total error change with respect to output.
        for j in range(t.nops):
            a = pderr(t.o[j], tgt[j])
            b = pdact(t.o[j])
            sum += a * b * t.w[t.xofs + j * t.nhid + i]
            # Correct weights in hidden to output layer.
            t.w[t.xofs + j * t.nhid + i] -= rate * a * b * t.h[i]
        # Correct weights in input to hidden layer.
        for j in range(t.nips):
            t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * inp[j]

def fprop(t:Tynn, inp:list[float]):
    """Performs forward propagation."""
    # Calculate hidden layer neuron values.
    for i in range(t.nhid):
        sum = float(0)
        for j in range(t.nips):
            sum += inp[j] * t.w[i * t.nips + j]
        t.h[i] = act(sum + t.b[0])
    # Calculate output layer neuron values.
    for i in range(t.nops):
        sum = float(0)
        for j in range(t.nhid):
            sum += t.h[j] * t.w[t.xofs + i * t.nhid + j]
        t.o[i] = act(sum + t.b[1])

from ctypes import CDLL
libc = CDLL("libc.so.6")
RAND_MAX=2147483647
def frand() -> float:
    return libc.rand() / RAND_MAX

def twrand(t:Tynn):
    for i in range(t.nw):
        t.w[i] = frand() - 0.5
    for i in range(t.nb):
        t.b[i] = frand() - 0.5

def xtpredict(t:Tynn, inp:list[float]) -> list[float]:
    """Returns an output prediction given an input."""
    fprop(t, inp)
    return t.o

def xttrain(t:Tynn, inp:list[float], tgt:list[float], rate:float) -> float:
    """Trains a tinn with an input and target output with a learning rate. Returns target to output error."""
    fprop(t, inp)
    bprop(t, inp, tgt, rate)
    return toterr(tgt, t.o, t.nops)

def xtbuild(nips:int, nhid:int, nops:int) -> Tynn:
    """Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs"""
    t = Tynn()
    t.nb = 2
    t.nw = nhid * (nips + nops)
    t.w = [float(0)] * t.nw
    t.xofs = nhid * nips
    t.b = [float(0)] * t.nb
    t.h = [float(0)] * nhid
    t.o = [float(0)] * nops
    t.nips = nips
    t.nhid = nhid
    t.nops = nops
    twrand(t)
    return t

def xtsave(t:Tynn, path:str):
    with open(path, "wt") as f:
        f.write(f"{t.nips:d} {t.nhid:d} {t.nops:d}\n")
        for i in range(t.nb):
            f.write(f"{t.b[i]:f}\n")
        for i in range(t.nw):
            f.write(f"{t.w[i]:f}\n")

def xtload(path:str) -> Tynn:
    try:
        with open(path, "rt") as f:
            l = f.readline().strip()
            # Load header.
            nips,nhid,nops=[int(e) for e in l.split(" ")]
            t = xtbuild(nips, nhid, nops)
            for i in range(t.nb):
                l = f.readline().strip()
                t.b[i] = float(l)
            for i in range(t.nw):
                l = f.readline().strip()
                t.w[i] = float(l)
            return t
    except:
        return None

def xtfree(tynn:Tynn):pass

def xtprint(arr:list[float], size:int):
    """Prints an array of floats. Useful for printing predictions."""
    for i in range(size):
        print(f"{arr[i]:.6f} ", end="")
    print()
