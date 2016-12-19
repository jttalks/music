# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 10:48:38 2016

@author: cferko
"""

import soundfile as sf
import requests, pickle, os, numpy as np
LOOKUP_DICT = pickle.load(open("file_lookup.pkl", "r"))
from matplotlib import mlab
import matplotlib.pyplot as plt

import IPython
from scipy.io.arff import loadarff
import sklearn.preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

features = loadarff("./genre_model.arff")
genres = ['blues',
          'classical',
          'country',
          'disco',
          'hiphop',
          'jazz',
          'metal',
          'pop',
          'reggae',
          'rock']

X = np.array(np.array([list(f)[:-1] for f in features[0]]))
y = [f[-1] for f in features[0]]
y = sklearn.preprocessing.LabelEncoder().fit_transform(y)

def get_features(genre, number):
    """Fetches the feature vector for a given genre and number
    """
    my_index = 100 * genres.index(genre) + (number - 1)
    return features[0][my_index]

def test_train_split():
    """Gets x and y, train and test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)
        
    return X_train, X_test, y_train, y_test
    

def standardize(old_rate, audio):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    if old_rate == 44100:
        return audio

    n = len(audio)
    upsample_factor = float(old_rate)/44100
    new_audio = np.interp(np.arange(0, n, upsample_factor),
                          np.arange(0, n, 1), audio)

    return new_audio

def listen(audio):
    return IPython.display.Audio(audio, rate = 44100)

def download_file(url):
    """Helper function to download a file with requests
    
    Args:
        url: a string pointing at a url with a downloadable resource
        
    Returns:
        the filename of the resource, on a successful return
    """
    local_filename = url.split('/')[-1].split("?")[0]
    # Hack to strip off the filename
    
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    
    return local_filename

def format_song_name(genre, number):
    """Helper function to get a filename
    
    Args:
        genre: a string, one of "blues", "classical", etc.
        number: an integer between 1 and 1000 (NOT zero-indexed)
        
    Returns:
        a string giving the filename for the specified song
    """
    zero_indexed_number = int(number) - 1
    padded_number = '{:05d}'.format(zero_indexed_number)
    
    name = genre + "." + padded_number + ".au"
    
    return name

def download_song(genre, number):
    """Fetches one of the 1000 songs
    
    Args:
        genre: a string, one of "blues", "classical", etc.
        number: an integer between 1 and 1000 (NOT zero-indexed)
        
    Returns:
        None (downloads the file to current directory)
    """
    name = format_song_name(genre, number)
    url = LOOKUP_DICT[name]
    download_file(url)
    
    return

def get_song(genre, number):
    """Student-facing convenience function to fetch a numpy array
    
    Args:
        genre: a string, one of "blues", "classical", etc.
        number: an integer between 1 and 1000 (NOT zero-indexed)
        
    Returns:
        None (downloads the file to current directory)
    """
    name = format_song_name(genre, number)
    if not os.path.isfile(name):
        download_song(genre, number)
        
    data, samplerate = sf.read(name)
    
    return standardize(samplerate, data)

def get_sample():
    closer = "https://www.dropbox.com/s/b7lcd4s7pvcroie/closer.wav?dl=1"
    
    if not os.path.isfile("closer.wav"):
        download_file(closer)
    d1, s1 = sf.read("closer.wav")

    return standardize(s1, d1)
    
def get_note():
    d, s = sf.read("concert_a.wav")

    return standardize(s, d)    

def get_chord():
    d, s = sf.read("major_triad.wav")

    return standardize(s, d) 

def get_scale():
    d, s = sf.read("c_major_scale.wav")

    return standardize(s, d) 

def ferko_zcr(my_sample):
    """one-liner trololololololo
    """
    return sum(my_sample[1:]*my_sample[:-1] < 0)/float(len(my_sample))    
    
def spectrogram(x, start = None, end = None, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=128,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs):

    fig = plt.figure()
    fig.set_size_inches(15.5, 8.5)
    ax = plt.gca()
    Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
         window, noverlap, pad_to, sides, scale_by_freq)

    if minfreq is not None and maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]

    Z = 10. * np.log10(Pxx)
    Z = np.flipud(Z)

    if xextent is None: 
        xextent = 0, np.amax(bins)
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = ax.imshow(Z, cmap, extent=extent, **kwargs)
    ax.axis('auto')
    
    ticks= list(plt.xticks()[0])
    plt.xticks(ticks, [t/16000.0 for t in ticks], rotation="horizontal")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Spectrogram")
    
    plt.xlim([0, xmax])

    return    
