import numpy as np

# Given a song, return its Short Time Fourier Transform
def STFT(song, FFT_SIZE = 1024, OVERLAP_FACTOR = 4):
    HOP_SIZE = FFT_SIZE // OVERLAP_FACTOR
    windowed_blocks = []
    padded = np.pad(song, (0, HOP_SIZE - len(song) % HOP_SIZE)) # pad to ensure a full last block
    start = 0 # index of start of the window
    window_fn = np.hanning(FFT_SIZE) # window function
    
    # sliding window to find all windowed blocks of signal
    while start+FFT_SIZE <= len(padded):
        windowed_blocks.append(padded[start:start+FFT_SIZE] * window_fn)
        start += HOP_SIZE
    windowed_blocks = np.array(windowed_blocks)
    return np.fft.rfft(windowed_blocks, n=FFT_SIZE)

# Given a STFT and the reconstruction length, conduct inverse Short Time Fourier Transform
def iSTFT(STFTx, min_len, FFT_SIZE = 1024, OVERLAP_FACTOR = 4):
    HOP_SIZE = FFT_SIZE // OVERLAP_FACTOR
    min_len += HOP_SIZE - min_len % HOP_SIZE # pad to full block size
    songx = np.zeros(min_len)
    FFT_SIZE = 1024
    OVERLAP_FACTOR = 4
    HOP_SIZE = FFT_SIZE // OVERLAP_FACTOR
    start = 0 # index of start of the window
    i = 0 # index of the block being processed
    window_fn = np.hanning(FFT_SIZE) # window function

    # sliding window to find all windowed blocks of signal
    while start+FFT_SIZE <= min_len:
        songx[start:start+FFT_SIZE] += np.fft.irfft(STFTx[i]) * window_fn
        start += HOP_SIZE
        i += 1
    return songx

# Given a complex signal, find its polar coordinates
def car2pol(sig):
    im, re = sig.imag, sig.real
    amp = np.sqrt(im**2 + re**2)
    angle = np.arctan2(im, re)
    return amp, angle

# Given polar coordinates, reconstruct the complex signal
def pol2car(amp, angle):
    re = amp * np.cos(angle)
    im = amp * np.sin(angle)
    return re + im*1j
