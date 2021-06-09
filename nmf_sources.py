from librosa.core import fft
from kl_nmf import kl_nmf
import soundfile as sf
import numpy as np
import librosa as lr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FFT_SIZE = 4096
HOPLEN = 512
COMPONENTS = 32 # NMF components
SOURCES = 4 # K in KMeans 

# Given a complex signal, find its polar coordinates
def car2pol(sig):
    im, re = sig.imag, sig.real
    amp = np.sqrt(im**2 + re**2)
    angle = np.arctan2(im, re)
    return amp, angle

def get_sources (mix, nsources=4, components=32, fft_size=4096, hoplen=512):
    # data preparation
    sources = np.zeros ((nsources, len (mix)))
    specgram = lr.stft(mix, n_fft=fft_size, hop_length=hoplen);
    A, Phi = car2pol(specgram)
    
    # NMF decomposition
    [W, H] = kl_nmf (A, components)
    
    # masking
    masks = np.zeros ((A.shape[0],A.shape[1],COMPONENTS),  dtype=complex)
    comps = np.zeros(masks.shape,  dtype=complex)
    for i in range (0, COMPONENTS):
        masks[:,:,i] = np.outer(W[:, i],H[i,:]) / np.dot(W,H)
        comps[:,:,i] =  masks[:,:,i] * specgram 

    # clustering
    scaler = StandardScaler()   
    scaled_features = scaler.fit_transform(H)
    clusters = KMeans (n_clusters=nsources).fit (scaled_features)

    # assignment and reconstruction
    for s in range (0, nsources): 
        print ("source ", s)
        comp = np.zeros (specgram.shape)
        for k in range (0, components):
            print ("comp ", k)
            if clusters.labels_[k] == s:
                comp = comp + comps[:, :, k]

        src = lr.istft (comp, hop_length=hoplen)                
        ml = min (len (src), len (mix))
        r = range(0, ml)
        sources[s, r] = src[r]
    return W, H, clusters.labels_, sources

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')

    mix, sr = sf.read ('samples/jungle1.wav')
    print ('total samples: ', len (mix))
    print ('sources      : ', SOURCES)
    print ('components   : ', COMPONENTS)

    W, H, labels, sources = get_sources (mix, nsources=SOURCES, components=COMPONENTS,
        fft_size=FFT_SIZE, hoplen=HOPLEN)        
    print ('W            : ', W.shape)
    print ('H            : ', H.shape)
    print ('labels       : ', labels)
    for s in range (SOURCES):
        sf.write ('source_' + str (s) + '.wav', sources[s], sr)

        

