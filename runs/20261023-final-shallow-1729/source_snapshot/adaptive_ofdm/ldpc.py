"""Sionna NR LDPC adapter with explicit LLR sign conversion.

Imports are lazy so numerical development checks do not need Sionna. This
adapter covers channel coding/rate matching, not an entire compliant NR PHY.
"""
import numpy as np


class NRLDPC:
    def __init__(self, k, n, iterations=12):
        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        self.torch = torch
        self.k, self.n, self.iterations = k, n, iterations
        self.encoder = LDPC5GEncoder(k=k, n=n, num_bits_per_symbol=2, device="cpu")
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=iterations,
                                    hard_out=True, return_infobits=True, device="cpu")

    def encode(self, bits):
        bits = np.asarray(bits)
        if bits.ndim != 2 or bits.shape[1] != self.k or not np.all((bits==0)|(bits==1)):
            raise ValueError("invalid information bits")
        with self.torch.no_grad():
            return self.encoder(self.torch.tensor(bits, dtype=self.torch.float32)).cpu().numpy().astype(np.uint8)

    def decode(self, llr):
        llr = np.asarray(llr)
        if llr.ndim != 2 or llr.shape[1] != self.n:
            raise ValueError("invalid LLR shape")
        # Sionna takes logits log(P(1)/P(0)); our PHY convention is the reverse.
        with self.torch.no_grad():
            return self.decoder(-self.torch.tensor(llr, dtype=self.torch.float32)).cpu().numpy().astype(np.uint8)
