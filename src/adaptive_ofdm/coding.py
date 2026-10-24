"""Small independent convolutional reference for physical-layer development.

Rate 1/2, constraint length 3, generators (7,5) octal, terminated in state 0.
This is NOT the main-paper LDPC implementation and is not NR compliant.
"""
import numpy as np


def transition(state, bit):
    register = (bit << 2) | state
    output = np.array([(register & 7).bit_count() % 2,
                       (register & 5).bit_count() % 2], np.uint8)
    return (bit << 1) | (state >> 1), output


def encode(information):
    information = np.asarray(information, np.uint8)
    if information.ndim != 2 or not np.all(information <= 1):
        raise ValueError("expected binary batch")
    out = np.empty((len(information), 2*(information.shape[1]+2)), np.uint8)
    for b, message in enumerate(information):
        state = 0
        for t, bit in enumerate(np.r_[message, [0, 0]]):
            state, output = transition(state, int(bit))
            out[b, 2*t:2*t+2] = output
        assert state == 0
    return out


def decode(llr):
    """Soft-input Viterbi; all LLRs use log(P(0)/P(1))."""
    llr = np.asarray(llr)
    if llr.ndim != 2 or llr.shape[1] % 2 or llr.shape[1] < 6:
        raise ValueError("invalid terminated codeword LLR shape")
    steps = llr.shape[1]//2
    result = np.empty((len(llr), steps-2), np.uint8)
    for b, row in enumerate(llr.reshape(len(llr), steps, 2)):
        metric = np.array([0., np.inf, np.inf, np.inf])
        predecessor = np.zeros((steps, 4), int)
        decisions = np.zeros((steps, 4), np.uint8)
        for t in range(steps):
            new = np.full(4, np.inf)
            for state in range(4):
                for bit in (0, 1):
                    dest, output = transition(state, bit)
                    branch = np.logaddexp(0, -(1-2*output.astype(float))*row[t]).sum()
                    value = metric[state]+branch
                    if value < new[dest]:
                        new[dest] = value
                        predecessor[t, dest] = state
                        decisions[t, dest] = bit
            metric = new
        state = 0
        message = np.zeros(steps, np.uint8)
        for t in range(steps-1, -1, -1):
            message[t] = decisions[t, state]
            state = predecessor[t, state]
        result[b] = message[:-2]
    return result
