import numpy as np
import pandas as pd
from scipy.signal import spectrogram


def a_weight(f):
    f2 = f ** 2
    return 1.2588966 * 148840000 * f2 * f2 / ((f2 + 424.36) * np.sqrt((f2 + 11599.29) *
                                                                      (f2 + 544496.41)) * (f2 + 148840000))

def transform_to_db(data: np.ndarray, reference: int = 10):
    return reference * np.log10(data + 1e-20)


def extract_spectrogram(Sxx: np.ndarray, f: np.ndarray, t_spec: np.ndarray, start: float, length:float):
    end = start + length

    # find the first bin ≥ start, and the first bin > end
    i0 = np.searchsorted(t_spec, start, side='left')
    i1 = np.searchsorted(t_spec, end, side='right')

    # clamp to valid range
    i0 = max(0, i0)
    i1 = min(len(t_spec), i1)
    if i0 >= i1:
        raise ValueError(f"No spectrogram bins in [{start}, {end}]")

    Sxx_seg = Sxx[:, i0:i1]
    t_seg = t_spec[i0:i1]

    return Sxx_seg, f, t_seg, i0, i1


def generate_spectrogram(data: pd.DataFrame, magnitude: str, freq: int, nperseg: int, noverlap: int, nfft: int):
    f, t_spec, Sxx = spectrogram(data[magnitude].values,
                                 fs=freq,
                                 window='hann',
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft
                                 )
    return f, t_spec, Sxx