# NMF-based source separation

Algorithm:
- STFT of input signal
- NMF decomposition in N components (KL)
- masking
- standardisation and dimensionality reduction of temporal activations
- clustering of temporal activations in K sources
- ISTFT reconstruction (phase can be recovered by iteration)

