'''
원본데이터는 https://physionet.org/content/siena-scalp-eeg/1.0.0/
visual-EEG (EO,EC)
sampling rate = 512Hz

edf 열고 sample 로 2-3개 channel만 열고 FFT하여 digitization 이전의 filtering 적용 여부를 살핀다.(비어있는 주파수 영역을 살핀다.)

입력으로 들어가는 channel 만 indexing (64ch -> 31ch)
resampling(1000Hz)
IIR  band pass filtering(0.1~70), attenuation dB 언급이 없음
이후 eeglab에서 ICA하고나서 (31ch-> 31ch)

random stride 작게주고 data_stride=2200fs로 자르기 (=2.2sec)

각각의 2200fs에 대하여

1. extract 'five statistical features'
2. resampling(100Hz)
---> 1,2를 동시에 한다. data_window=100fs, data_stride=50fs로 하면 될듯(paper에는 자세한 언급이 없다)
---> 결과가 담긴 변수는 apply_5f

이후 KPCA(eigen value 큰 30개만 남긴다.)

orignina data에 delta(f(t) - f(t-1)), delta-delta 적용하여 *3한다.
(결국 grad를 구하는건데 edge detector 와 유사하다고 보면된다.)

total EEG data dim =90
--->최종data 변수는 final_data

total shape=(90,44)

정규화(scipy.stats.zscore)에 대한 언급은 없어 그냥 안하기로...   ----> train_data_make(preICH)에서 하자 by scipy.stats.zscore

'''

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as ss
import os
import EDFlib
import math
from sklearn.decomposition import KernelPCA




PATH="raw_data/preICA/"
save_PATH="train_data/"
len_data=2200   # 2second (128Hz)
sampling_stride=50  # 의미는 없다
file_name=["preICA0.edf", "preICA1.edf"]  # 파일이 많아지면 for문으로 만들자
len_channel=31
fs=1000


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    '''
    [low, high]
    The critical frequency or frequencies. 
    For lowpass and highpass filters, Wn is a scalar; 
    for bandpass and bandstop filters, Wn is a length-2 sequence.
    For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”). 
    For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, 
    so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.) 
    For analog filters, Wn is an angular frequency (e.g. rad/s).
    '''

    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y




def resample(data, num_channel, sampling_fs, resampling_fs):
    data_len=data.shape[1]
    new_data= np.zeros([data.shape[0], int(np.round(data_len * resampling_fs/sampling_fs))])
    for j in range(num_channel):
        new_data[j]=signal.resample(data[j], int(np.round(data_len * resampling_fs/sampling_fs)))

    return new_data




def cheby2_lowpass_filter(data, highcut, fs, order=71, riple=20):
    nyq = 0.5 * fs
    b, a = signal.cheby2(order, riple, highcut / nyq, btype='lowpass')
    output = signal.filtfilt(b, a, data)

    return output




def cheby2_highpass_filter(data, lowcut, fs, order=24, riple=20):
    nyq = 0.5 * fs
    b, a = signal.cheby2(order, riple, lowcut / nyq, btype='high')
    output = signal.filtfilt(b, a, data)

    return output



def notch_pass_filter(data, center, interval=2, fs=1000):   # interval = 대역폭 (좌우로 얼마나 지울지) sharp 하게 적용하자, 차라리 for 문으로 49,50,51을 모두 sharp하게 지우자
    b, a= signal.iirnotch(center, center/interval,fs)
    filtered_data=signal.lfilter(b,a,data)

    return filtered_data



# len_data=100
def root_mean_square(data):
    square_sum=0
    for i in range(len(data)):
        square=(data[i])**2
        square_sum+=square
    root_mean_square=math.sqrt(square_sum)

    return root_mean_square





def zero_crossing_rate(data):
    zero_rate=0
    for i in range(len(data)-1):
        if data[i]*data[i+1]<0:
            zero_rate+=1
    return zero_rate





def moving_window_average(data):
    sum=0
    for i in range(len(data)):
        sum+=data[i]
    average=sum/len(data)
    return average



##########################################################################################
'''def kurtosis inner function'''
##########################################################################################
def mean(inp):
    result = 0
    len_inp = len(inp)
    for i in inp:
        result += i
    result = result / len_inp
    return result

def var(inp):
    result = 0
    len_inp = len(inp)
    for i in inp:
        result += (i - mean(inp)) ** 2
    result = result / len_inp
    return result

def sqrt(inp):
    result = inp/2
    for i in range(30):
        result = (result + (inp / result)) / 2
    return result

def std(inp):
    result = sqrt(var(inp))
    return result

##########################################################################################




def kurtosis(inp):
    # 길이
    len_inp = len(inp)
    result = 0
    for i in inp:
        result += ((i - mean(inp)) / std(inp)) ** 4
    result = (result / len_inp) - 3
    return result



# https://github.com/raphaelvallat/entropy/blob/master/entropy/entropy.py
def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False,
                     axis=-1):
    """Spectral Entropy.
    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float
        Sampling frequency, in Hz.
    method : str
        Spectral estimation method:
        * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
        * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
    nperseg : int or None
        Length of each FFT segment for Welch method.
        If None (default), uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).
    Returns
    -------
    se : float
        Spectral Entropy
    Notes
    -----
    Spectral Entropy is defined to be the Shannon entropy of the power
    spectral density (PSD) of the data:
    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]
    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.
    References
    ----------
    - Inouye, T. et al. (1991). Quantification of EEG irregularity by
      use of the entropy of the power spectrum. Electroencephalography
      and clinical neurophysiology, 79(3), 204-210.
    - https://en.wikipedia.org/wiki/Spectral_density
    - https://en.wikipedia.org/wiki/Welch%27s_method
    Examples
    --------
    Spectral entropy of a pure sine using FFT
    >>> import numpy as np
    >>> import entropy as ent
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ent.spectral_entropy(x, sf, method='fft'), 2)
    0.0
    Spectral entropy of a random signal using Welch's method
    >>> np.random.seed(42)
    >>> x = np.random.rand(3000)
    >>> ent.spectral_entropy(x, sf=100, method='welch')
    6.980045662371389
    Normalized spectral entropy
    >>> ent.spectral_entropy(x, sf=100, method='welch', normalize=True)
    0.9955526198316071
    Normalized spectral entropy of 2D data
    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> np.round(ent.spectral_entropy(x, sf=100, normalize=True), 4)
    array([0.9464, 0.9428, 0.9431, 0.9417])
    Fractional Gaussian noise with H = 0.5
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9505
    Fractional Gaussian noise with H = 0.9
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.8477
    Fractional Gaussian noise with H = 0.1
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9248
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = signal.periodogram(x, sf, axis=axis)

    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm+(1e-100))).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis]+(1e-100))
    return se



def delta(data):
    length=len(data)
    delta_data=np.zeros(length)
    for i in range(length):
        if i==0:
            delta_data[i]=0
        else:
            delta_data[i] = data[i]-data[i-1]
    return delta_data






len_window=100
len_window_stride=50
final_data_shape=[90,43]
num_window=43

def make():

    for i in range(len(file_name)):
        name=file_name[i]

        data = mne.io.read_raw_edf(os.path.join(PATH, name))
        new_data = data.get_data()
        num_channel, data_len =new_data.shape

        j=len_data
        m=-1

        with open(save_PATH + 'EEG_filenames/' + 'EEG_filenames.txt', 'a') as f:

            while True:
                k = np.random.randint(5, 15)
                j += sampling_stride + k +len_data
                if j >= new_data.shape[1]:    #(9760?)   ---> EEG_segmented는 총 #3가 나온다
                    break
                EEG_segmented = new_data[:, j - len_data*2 - sampling_stride - k:j - len_data - sampling_stride - k]
                print('first')
                print(EEG_segmented.shape)  # (31,2200)

                n=len_window
                apply_5f=np.zeros((EEG_segmented.shape[0]*5, int(EEG_segmented.shape[1]/50)-int(1)))
                final_data = np.zeros((final_data_shape[0], final_data_shape[1]))
                print(apply_5f.shape)    # (155,44)
                m += 1




                # apply_5f 를 만들어준다 최종 shape (155,43)
                for u in range(num_window):   # range(43)
                    n += len_window_stride
                    window = EEG_segmented[:, n - len_window - len_window_stride:n - len_window_stride]
                    print(window.shape)     # (31,100)
                    for p in range(window.shape[0]):   #window.shape[0] = 31

                        apply_5f[5*p, u] = root_mean_square(window[p,:])
                        apply_5f[5*p + 1, u] = zero_crossing_rate(window[p, :])
                        apply_5f[5*p + 2, u] = moving_window_average(window[p, :])
                        apply_5f[5*p + 3, u] = kurtosis(window[p, :])
                        apply_5f[5*p + 4, u] = spectral_entropy(window[p, :], 1000, method='fft')

                #kpca
                apply_5f_new=np.transpose(apply_5f)
                print(apply_5f_new.shape)   # (43, 155)   (N_samples, n_features)
                kpca = KernelPCA(n_components=30, kernel='linear', gamma='none')
                post_kpca = kpca.fit_transform(apply_5f_new)
                post_kpca=np.transpose(post_kpca)
                print(post_kpca.shape)   # (30,43)

                # delta, delta-delta
                for r in range(post_kpca.shape[0]):
                    final_data[3 * r, :] = post_kpca[r, :]
                    final_data[3 * r + 1, :] = delta(post_kpca[r, :])
                    final_data[3 * r + 2, :] = delta(delta(post_kpca[r, :]))
                print('final_data shape')

                print(final_data.shape)



                np.save(save_PATH + 'EEG_datas/' + 'EEG_{0}_{1}.npy'.format(i,m), final_data)
                f.write('EEG_{0}_{1}.npy\n'.format(i,m))




if __name__ == '__main__':
    make()



