'''

RP_extract: Rhythm Patterns Audio Feature Extractor

@author: 2014-2015 Alexander Schindler, Thomas Lidy


Re-implementation by Alexander Schindler of RP_extract for Matlab
Matlab version originally by Thomas Lidy, based on Musik Analysis Toolbox by Elias Pampalk
( see http://ifs.tuwien.ac.at/mir/downloads.html )

Main function is rp_extract. See function definition and description for more information,
or example usage in main function.

Note: All required functions are provided by the two main scientific libraries numpy and scipy.

Note: In case you alter the code to use transform2mel, librosa needs to be installed: pip install librosa
'''


import numpy as np
from scipy.fftpack import fft


# UTILITY FUNCTIONS


def nextpow2(num):
    '''NextPow2

    find the next highest number to the power of 2 to a given number
    and return the exponent to 2
    (analogously to Matlab's nextpow2() function)
    '''

    n = 2
    i = 1
    while n < num:
        n *= 2 
        i += 1
    return i



# FFT FUNCTIONS

def periodogram(x,win,Fs=None,nfft=1024):
    ''' Periodogram

    Periodogram power spectral density estimate
    Note: this function was written with 1:1 Matlab compatibility in mind.

    The number of points, nfft, in the discrete Fourier transform (DFT) is the maximum of 256 or the next power of two greater than the signal length.

    :param x: time series data (e.g. audio signal), ideally length matches nfft
    :param win: window function to be applied (e.g. Hanning window). in this case win expects already data points of the window to be provided.
    :param Fs: sampling frequency (unused)
    :param nfft: number of bins for FFT (ideally matches length of x)
    :return: Periodogram power spectrum (np.array)
    '''


    #if Fs == None:
    #    Fs = 2 * np.pi         # commented out because unused
   
    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U
    
    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    
    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        #select = np.arange(nfft/2+1);    # EVEN
        #P = P[select,:]         # Take only [0,pi] or [0,pi) # TODO: why commented out?
        P[1:-2] = P[1:-2] * 2

    P = P / (2 * np.pi)

    return P




def calc_spectrogram(wavsegment,fft_window_size,fft_overlap = 0.5,real_values=True):

    ''' Calc_Spectrogram

    calculate spectrogram using periodogram function (which performs FFT) to convert wave signal data
    from time to frequency domain (applying a Hanning window and (by default) 50 % window overlap)

    :param wavsegment: audio wave file data for a segment to be analyzed (mono (i.e. 1-dimensional vector) only
    :param fft_window_size: windows size to apply FFT to
    :param fft_overlap: overlap to apply during FFT analysis in % fraction (e.g. default = 0.5, means 50% overlap)
    :param real_values: if True, return real values by taking abs(spectrogram), if False return complex values
    :return: spectrogram matrix as numpy array (fft_window_size, n_frames)
    '''

    # hop_size (increment step in samples, determined by fft_window_size and fft_overlap)
    hop_size = int(fft_window_size*(1-fft_overlap))

    # this would compute the segment length, but it's pre-defined above ...
    # segment_size = fft_window_size + (frames-1) * hop_size
    # ... therefore we convert the formula to give the number of frames needed to iterate over the segment:
    n_frames = int((wavsegment.shape[0] - fft_window_size) / hop_size + 1)
    # n_frames_old = wavsegment.shape[0] / fft_window_size * 2 - 1  # number of iterations with 50% overlap

    # TODO: provide this as parameter for better caching?
    han_window = np.hanning(fft_window_size) # verified

    # initialize result matrix for spectrogram
    spectrogram = np.zeros((fft_window_size, n_frames), dtype=np.complex128)

    # start index for frame-wise iteration
    ix = 0

    for i in range(n_frames): # stepping through the wave segment, building spectrum for each window
        spectrogram[:,i] = periodogram(wavsegment[ix:ix+fft_window_size], win=han_window,nfft=fft_window_size)
        ix = ix + hop_size

        # NOTE: tested scipy periodogram BUT it delivers totally different values AND takes 2x the time of our periodogram function (0.13 sec vs. 0.06 sec)
        # from scipy.signal import periodogram # move on top
        #f,  spec = periodogram(x=wavsegment[idx],fs=samplerate,window='hann',nfft=fft_window_size,scaling='spectrum',return_onesided=True)

    if real_values: spectrogram = np.abs(spectrogram)

    return (spectrogram)



# Transform 2 Mel Scale: NOT USED by rp_extract, but included for testing purposes or for import into other programs

def transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands = 80,freq_min = 0,freq_max = None):
    '''Transform to Mel

    convert a spectrogram to a Mel scale spectrogram by grouping original frequency bins
    to Mel frequency bands (using Mel filter from Librosa)

    Parameters
    spectrogram: input spectrogram
    samplerate: samplerate of audio signal
    fft_window_size: number of time window / frequency bins in the FFT analysis
    n_mel_bands: number of desired Mel bands, typically 20, 40, 80 (max. 128 which is default when 'None' is provided)
    freq_min: minimum frequency (Mel filters will be applied >= this frequency, but still return n_meld_bands number of bands)
    freq_max: cut-off frequency (Mel filters will be applied <= this frequency, but still return n_meld_bands number of bands)

    Returns:
    mel_spectrogram: Mel spectrogram: np.array of shape(n_mel_bands,frames) maintaining the number of frames in the original spectrogram
    '''

    from librosa.filters import mel

    # Syntax: librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False)
    mel_basis = mel(samplerate,fft_window_size, n_mels=n_mel_bands,fmin=freq_min,fmax=freq_max)

    freq_bin_max = mel_basis.shape[1] # will be fft_window_size / 2 + 1

    # IMPLEMENTATION WITH FOR LOOP
    # initialize Mel Spectrogram matrix
    #n_mel_bands = mel_basis.shape[0]  # get the number of bands from result in case 'None' was specified as parameter
    #mel_spectrogram = np.empty((n_mel_bands, frames))

    #for i in range(frames): # stepping through the wave segment, building spectrum for each window
    #    mel_spectrogram[:,i] = np.dot(mel_basis,spectrogram[0:freq_bin_max,i])

    # IMPLEMENTATION WITH DOT PRODUCT (15% faster)
    # multiply the mel filter of each band with the spectogram frame (dot product executes it on all frames)
    # filter will be adapted in a way so that frequencies beyond freq_max will be discarded
    mel_spectrogram = np.dot(mel_basis,spectrogram[0:freq_bin_max,:])
    return (mel_spectrogram)
