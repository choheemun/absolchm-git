'''
원본데이터는 https://physionet.org/content/siena-scalp-eeg/1.0.0/
visual-EEG (EO,EC)
sampling rate = 512Hz

edf 열고 sample 로 2-3개 channel만 열고 FFT하여 digitization 이전의 filtering 적용 여부를 살핀다.(비어있는 주파수 영역을 살핀다.)

입력으로 들어가는 channel 만 indexing (64ch -> 32ch)
resampling(1000Hz)
IIR  band pass filtering(0.1~70), attenuation dB 언급이 없음
이후 eeglab에서 ICA하고나서 (31ch-> 31ch)

random stride 작게주고 data_stride=2200fs로 자르기 (=2.2sec)

각각의 2200fs에 대하여

1. extract 'five statistical features'
2. resampling(100Hz)
---> 1,2를 동시에 한다. data_window=100fs, data_stride=10fs로 하면 될듯(paper에는 자세한 언급이 없다)

이후 KPCA(eigen value 큰 30개만 남긴다.)

total EEG data dim =90

정규화(scipy.stats.zscore)에 대한 언급은 없어 그냥 안하기로...

'''

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as ss
import os
import EDFlib



PATH="raw_data/"
channel_sort=[1, 3, 5, 7, 9, 13, 15, 17, 19, 21, 22, 24, 30, 32, 34, 36, 38, 39, 40, 41, 42, 45, 46, 47, 49, 51, 53, 55, 61, 62, 63]
len_data=2200   # 2second (128Hz)
sampling_stride=100  # 의미는 없다
file_name=["S001R01.edf", "S001R02.edf"]
cut_list=[0.1,70]   # butterworth band pass filter lowcut & highcut
len_channel=31
notch_list=[59,60,61]
notch_filtering_bandwidth=2
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




ch_names = []
for i in range(31):
    i = str(i + 1)
    ch_names.append(i)
sfreq = 100
ch_types = 'eeg'



def make():

    for i in range(2):
        name=file_name[i]

        data = mne.io.read_raw_edf(os.path.join(PATH, name))
        raw_data = data.get_data()
        num_channel, data_len =raw_data.shape
        print(raw_data.shape)     #

        new_raw_data = np.zeros((len(channel_sort), data_len),dtype=float)
        for l in range(len(channel_sort)):
            new_raw_data[l]=raw_data[channel_sort[l]]

        num_channel, data_len = new_raw_data.shape
        new_raw_data2 = np.zeros((num_channel, data_len),dtype=float)
        for j in range(num_channel):
            new_raw_data2[j]=ss.zscore(new_raw_data[j])         # normalizing

        print(new_raw_data2.shape)
        for k in range(len(notch_list)):
            new_raw_data = notch_pass_filter(new_raw_data2, notch_list[k], notch_filtering_bandwidth, fs)

        new_raw_data = butter_bandpass_filter(new_raw_data, cut_list[0], cut_list[1], fs, order=2)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        new_raw_data = mne.io.RawArray(new_raw_data, info=info)
        mne.export.export_raw('raw_data/preICA/preICA{0}.edf'.format(i), new_raw_data, fmt='edf')







if __name__ == '__main__':
    make()



