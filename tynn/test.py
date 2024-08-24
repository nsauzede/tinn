#!/usr/bin/env python3

import sys
from dataclasses import dataclass, field

from tynn import *

# Data object.
@dataclass
class Data:
    # 2D floating point array of input.
    inp:list[list[float]]=field(default_factory=list)
    # 2D floating point array of target.
    tgt:list[list[float]]=field(default_factory=list)
    # Number of inputs to neural network.
    nips:int=0
    # Number of outputs to neural network.
    nops:int=0
    # Number of rows in file (number of sets for neural network).
    rows:int=0

from ctypes import CDLL
libc = CDLL("libc.so.6")
def shuffle(d:Data):
    """Randomly shuffles a data object."""

    for a in range(d.rows):
        b = libc.rand() % d.rows
        ot = d.tgt[a]
        it = d.inp[a]
        # Swap output.
        d.tgt[a] = d.tgt[b]
        d.tgt[b] = ot
        # Swap input.
        d.inp[a] = d.inp[b]
        d.inp[b] = it

def build(path:str, nips:int, nops:int) -> Data:
    """Parses file from path getting all inputs and outputs for the neural network. Returns data object."""

    try:
        with open(path, "rt") as f:
            lines = f.readlines()
    except:
        print(f"Could not open {path}")
        print("Get it from the machine learning database: ")
        print("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data")
        sys.exit(1)
    data = Data()
    for l in lines:
        spl = l.strip().split(" ")
        inp = [float(f) for f in spl[:nips]]
        tgt = [float(f) for f in spl[nips:]]
        data.inp += [inp]
        data.tgt += [tgt]
    data.rows = len(lines)
    data.nips = nips
    data.nops = nops
    return data

def inp_disp(inp:list[float], w:int, h:int):
    for j in range(h):
        for i in range(w):
            print('#' if inp[j * w + i]>0 else ' ', end="")
        print()

def pd_disp(pd:list[float]):
    for i,p in enumerate(pd):
        if p>0.90:
            print(f"Predicted Handwriting to be an '{i}' (score: ~{100*p:.2f}%)")
            return
    print("Couldn't reliably predict Handwriting ??")

def main() -> int:
    """Learns and predicts hand written digits with 98% accuracy."""

    # Tinn does not seed the random number generator.
    libc.srand(libc.time(0))
    #libc.srand(0)
    # Input and output size is harded coded here as machine learning
    # repositories usually don't include the input and output size in the data itself.
    nips = 256
    nops = 10
    # Hyper Parameters.
    # Learning rate is annealed and thus not constant.
    # It can be fine tuned along with the number of hidden layers.
    # Feel free to modify the anneal rate.
    # The number of iterations can be changed for stronger training.
    rate = 1.0
    nhid = 28
    anneal = 0.99
    #iterations = 128   # very slow with Python
    iterations = 10     # good enough ? 10.99 for 127
    #iterations = 16     # good enough ? 10.99 for 127
    # Load training set.
    data = build("semeion.data", nips, nops)
    # This is how you load the neural network from disk.
    loaded = xtload("saved.tinn")
    if loaded==None:
        print("Failed to load saved? Training now..")
        # Train, baby, train.
        tynn = xtbuild(nips, nhid, nops)
        for i in range(iterations):
            shuffle(data)
            error = 0.0
            for j in range(data.rows):
                inp = data.inp[j]
                tgt = data.tgt[j]
                error += xttrain(tynn, inp, tgt, rate)
            print(f"error {error / data.rows:.12f} :: learning rate {rate:f}")
            rate *= anneal
        # This is how you save the neural network to disk.
        xtsave(tynn, "saved.tinn")
        loaded = tynn
    else:
        shuffle(data)
    nmiss=0
    gmini=100
    for idx in range(data.rows):
        inp = data.inp[idx]
        tgt = data.tgt[idx]
        pd = xtpredict(loaded, inp)
        maxi=0
        for pred in pd:
            if pred>maxi:
                maxi=pred
        if maxi<gmini:
            gmini = maxi
        if maxi<0.90:
            #print(f"Max pred only {100*maxi:.2f}% for idx={idx} ?")
            # Prints target.
            #xtprint(tgt, data.nops)
            # Prints prediction.
            #xtprint(pd, data.nops)
            # Display input.
            #inp_disp(inp, 16, 16)
            # Display prediction.
            #pd_disp(pd)
            nmiss += 1
    print(f"nmiss={nmiss} gmini={gmini*100:.2f}%")
    # Now we do a prediction with the neural network we loaded from disk.
    # Ideally, we would also load a testing set to make the prediction with,
    # but for the sake of brevity here we just reuse the training set from earlier.
    # One data set is picked at random (zero index of input and target arrays is enough
    # as they were both shuffled earlier).
    idx = libc.rand() % data.rows
    print(f"Using index={idx}")
    inp = data.inp[idx]
    tgt = data.tgt[idx]
    pd = xtpredict(loaded, inp)
    # Prints target.
    xtprint(tgt, data.nops)
    # Prints prediction.
    xtprint(pd, data.nops)
    # Display input.
    inp_disp(inp, 16, 16)
    # Display prediction.
    pd_disp(pd)
    # All done. Let's clean up.
    xtfree(loaded)
    return 0

if __name__ == '__main__':
    sys.exit(main())
