"""Time-domain CP-OFDM reference, with explicit energy and information boundaries.

All FFTs on waveforms are unitary. Channel H uses the ordinary FFT, so after
CP removal Y[k] = H[k] X[k] + W[k]. Noise variance is E[|W[k]|**2].
LLRs are log(P(bit=0)/P(bit=1)); positive means bit zero.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Numerology:
    fft_size: int = 64
    cp_length: int = 16
    symbols: int = 8
    guard: int = 4
    pilot_spacing: int = 4

    def __post_init__(self):
        if self.fft_size < 16 or self.fft_size % 2:
            raise ValueError("fft_size must be even and at least 16")
        if not 1 <= self.cp_length < self.fft_size:
            raise ValueError("cp_length must be in [1, fft_size)")
        if not 1 <= self.guard < self.fft_size // 4:
            raise ValueError("invalid guard")
        if self.symbols < 1 or self.pilot_spacing < 2:
            raise ValueError("invalid symbol count or pilot spacing")

    @property
    def active(self):
        """Sorted signed frequencies, omitting DC; edge guards on each side."""
        return np.concatenate((np.arange(-self.fft_size//2+self.guard, 0),
                               np.arange(1, self.fft_size//2-self.guard)))

    @property
    def pilots(self):
        a = self.active
        return a[np.unique(np.r_[np.arange(0, len(a), self.pilot_spacing), len(a)-1])]

    @property
    def data(self):
        return np.setdiff1d(self.active, self.pilots)

    @property
    def coded_bits(self):
        return self.symbols * len(self.data) * 2


def qpsk_map(bits):
    bits = np.asarray(bits)
    if bits.shape[-1] % 2 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("QPSK requires binary pairs")
    b = bits.reshape(*bits.shape[:-1], -1, 2).astype(float)
    return ((1-2*b[..., 0]) + 1j*(1-2*b[..., 1])) / np.sqrt(2)


def qpsk_llr(equalized, effective_variance):
    v = np.maximum(np.asarray(effective_variance), 1e-12)
    out = np.stack((2*np.sqrt(2)*equalized.real/v,
                    2*np.sqrt(2)*equalized.imag/v), axis=-1)
    return out.reshape(*out.shape[:-2], -1)


def make_grid(bits, cfg):
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != cfg.coded_bits:
        raise ValueError("bits must have shape (batch, coded_bits)")
    grid = np.zeros((len(bits), cfg.symbols, cfg.fft_size), complex)
    grid[..., cfg.data % cfg.fft_size] = qpsk_map(bits).reshape(len(bits), cfg.symbols, -1)
    grid[..., cfg.pilots % cfg.fft_size] = 1 + 0j
    return grid


def modulate(grid, cfg):
    x = np.fft.ifft(grid, norm="ortho", axis=-1)
    return np.concatenate((x[..., -cfg.cp_length:], x), axis=-1).reshape(len(x), -1)


def demodulate(waveform, cfg):
    x = waveform.reshape(len(waveform), cfg.symbols, cfg.fft_size+cfg.cp_length)
    return np.fft.fft(x[..., cfg.cp_length:], norm="ortho", axis=-1)


def expected_frame_energy(cfg):
    """Exact expectation over independent zero-mean unit-energy data symbols.

    All pilots equal one. Their coherent cross terms affect the energy in the
    CP subset, although full-symbol Parseval energy remains the tone count.
    """
    pilot_grid=np.zeros(cfg.fft_size,complex)
    pilot_grid[cfg.pilots%cfg.fft_size]=1
    pilot_time=np.fft.ifft(pilot_grid,norm="ortho")
    cp_energy=cfg.cp_length*len(cfg.data)/cfg.fft_size+np.sum(abs(pilot_time[-cfg.cp_length:])**2)
    return float(cfg.symbols*(len(cfg.active)+cp_energy))


def apply_channel(waveform, taps, cfg):
    if taps.ndim not in (2,3) or len(taps) != len(waveform):
        raise ValueError("taps must have shape (batch, channel_length[, samples])")
    if taps.shape[1] - 1 > cfg.cp_length:
        raise ValueError("reference channel memory exceeds CP")
    if taps.ndim==2:
        return np.stack([np.convolve(x, h)[:len(x)] for x, h in zip(waveform, taps)])
    if taps.shape[2]!=waveform.shape[1]:
        raise ValueError("time-varying taps must cover each output sample")
    result=np.zeros(waveform.shape,dtype=np.result_type(waveform,taps))
    for lag in range(taps.shape[1]):
        result[:,lag:]+=taps[:,lag,lag:]*waveform[:,:waveform.shape[1]-lag]
    return result


def rayleigh_taps(rng, batch, length=5):
    """Independent static-per-frame taps with expected total energy one.

    Do not normalize each realization: that would remove fading in magnitude.
    """
    if length < 1:
        raise ValueError("length must be positive")
    power = np.exp(-np.arange(length)/2)
    power /= power.sum()
    return (rng.standard_normal((batch, length)) +
            1j*rng.standard_normal((batch, length))) * np.sqrt(power/2)


def markov_bursts(rng, batch, samples, occupancy, mean_length):
    """Stationary two-state chain with prescribed mean occupied run length."""
    if not 0 <= occupancy < 1 or mean_length < 1:
        raise ValueError("invalid burst model")
    p10 = 1/mean_length
    p01 = occupancy*p10/(1-occupancy)
    if p01 > 1:
        raise ValueError("occupancy and mean_length imply invalid transition")
    state = rng.random(batch) < occupancy
    mask = np.empty((batch, samples), bool)
    for t in range(samples):
        mask[:, t] = state
        u = rng.random(batch)
        state = np.where(state, u >= p10, u < p01)
    return mask


@dataclass
class Observation:
    """Only information potentially available at the receiver.

    nominal_noise_variance is an oracle diagnostic input until a practical
    estimator is supplied. Do not feed it into the proposed gate by default.
    """
    waveform: np.ndarray
    grid: np.ndarray
    nominal_noise_variance: np.ndarray


@dataclass
class Truth:
    """Training/evaluation labels; never accepted by a deployed gate."""
    coded_bits: np.ndarray
    taps: np.ndarray
    burst_mask: np.ndarray


def transmit(bits, cfg, rng, ebno_db, information_bits=None, taps=None,
             occupancy=0.0, mean_burst_length=4.0, inr_db=15.0):
    """Inject disturbances BEFORE CP removal and FFT.

    Eb includes expected transmitted pilot and CP energy / information bits.
    Interference power (when active) = nominal AWGN variance * 10**(INR/10).
    SNR/EbNo describes background noise only, not total interference.
    """
    bits = np.asarray(bits)
    grid = make_grid(bits, cfg)
    x = modulate(grid, cfg)
    k = cfg.coded_bits if information_bits is None else information_bits
    if not 0 < k <= cfg.coded_bits:
        raise ValueError("invalid information bit count")
    # Fixed expected energy avoids making the noise variance depend on payload.
    eb = expected_frame_energy(cfg)/k
    n0 = np.broadcast_to(eb / 10**(np.asarray(ebno_db)/10), (len(bits),)).copy()
    if taps is None:
        taps = rayleigh_taps(rng, len(bits))
    clean = apply_channel(x, taps, cfg)
    noise = (rng.standard_normal(x.shape)+1j*rng.standard_normal(x.shape))*np.sqrt(n0[:, None]/2)
    mask = markov_bursts(rng, len(bits), x.shape[1], occupancy, mean_burst_length)
    impulse = (rng.standard_normal(x.shape)+1j*rng.standard_normal(x.shape))*np.sqrt(n0[:, None]*10**(inr_db/10)/2)
    y = clean + noise + mask*impulse
    return Observation(y, demodulate(y, cfg), n0), Truth(bits, taps, mask)


def ls_channel(grid, cfg):
    """Per-symbol LS pilots and linear interpolation in signed-frequency order."""
    h_p = grid[..., cfg.pilots % cfg.fft_size]  # known pilots equal 1
    h = np.empty((*grid.shape[:2], len(cfg.data)), complex)
    for b in range(len(grid)):
        for s in range(cfg.symbols):
            h[b, s] = (np.interp(cfg.data, cfg.pilots, h_p[b, s].real) +
                       1j*np.interp(cfg.data, cfg.pilots, h_p[b, s].imag))
    return h


def equalize_llr(grid, channel_data, noise_variance, cfg):
    """Scalar coherent QPSK reference. Noise/CSI knowledge must be labelled."""
    h = channel_data
    power = np.maximum(np.abs(h)**2, 1e-12)
    z = grid[..., cfg.data % cfg.fft_size]*h.conj()/power
    var = np.asarray(noise_variance)[:, None, None]/power
    return qpsk_llr(z, var).reshape(len(grid), -1)
