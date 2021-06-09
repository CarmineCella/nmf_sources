# NMF-based source separation

Algorithm:
- STFT of input signal
- NMF decomposition in N components (KL)
- masking
- clustering in K sources
- standardisation  of temporal activations
- clustering of temporal activations in K sources
- ISTFT reconstruction

