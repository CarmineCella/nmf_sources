# NMF-based source separation

Algorithm:
- STFT of input signal
- NMF decomposition in N components (KL)
- masking
- clustering in K sources
- ISTFT reconstruction (phase can be recovered by iteration)

