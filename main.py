import bisect
import numpy as np
# from ecgdetectors import Detectors
import mne
import matplotlib
from pathlib import Path
from scipy.signal import butter, iirnotch, lfilter, filtfilt
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
import neurokit2 as nk

matplotlib.pyplot.figure
matplotlib.lines
matplotlib.lines.Line2D

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_highpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    return filter_signal(data=data, cutoff=cutoff, sample_rate=sample_rate, filtertype='notch')
def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass', return_top=False):
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
    cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: \
    lowpass, highpass, bandpass, and notch' % filtertype)

    filtered_data = filtfilt(b, a, data)

    if return_top:
        return np.clip(filtered_data, a_min=0, a_max=None)
    else:
        return filtered_data


# =====================================================
"Reading the EDF file"
# =====================================================

path = "C:\\Users\\Manef\\Desktop\\data\\siena-scalp-eeg-database-1.0.0\\siena-scalp-eeg-database-1.0.0\\PN06\\"
ID = "PN06-2"
fs = 512
file_path = path + ID + ".edf"
edf = read_raw_edf(file_path, preload=False, stim_channel=None, verbose=False)
xx = edf.ch_names
index = xx.index("EEG T5")
fs = edf.info['sfreq']
fs = int(fs)
signal_input = edf[index]
signal = signal_input[0]
signal_input = signal[0]

# =====================================================
"extracting the needed part of the signal to work with"
# =====================================================

end = (fs * 60 * 60 * 2) + (fs * 60 * 40) + (20 * fs)
pre_ictal = (fs * 60 * 60 * 1) + (fs * 60 * 6) + (20 * fs)
signal_input = signal_input[pre_ictal:end]

# ======================================================
"In this part we delete the unwanted part of the input ECG signal where we delete this segment from [fromm[i]- to[i]] "
# =====================================================

fromm = [59000, 72000]
to = [59900, 72900]
for i in range (len(fromm)):

    signal_input_1 = signal_input[0:fromm[i]]
    signal_input1 = signal_input[to[i]:len(signal_input)]
    signal_input_1 = np.append(signal_input_1, signal_input1)
    signal_input = signal_input_1

# =====================================================
"signal filtering"
# =====================================================

signal_output = filter_signal(data=signal_input, cutoff=60, sample_rate=fs, order=2, filtertype="notch")
signal_output = filter_signal(data=signal_output, cutoff=20, sample_rate=fs, order=2, filtertype="highpass")
signal_output = filter_signal(data=signal_output, cutoff=10, sample_rate=fs, order=2, filtertype="lowpass")

# =====================================================
"R peaks detection step"
# =====================================================

print('signal duration\t\t', ((len(signal_output) / fs) / 60))
print('signal duration\t\t', len(signal_input))
print('start to compute the entropy features:')
HRV = []

r_peaks = nk.ecg_peaks(signal_output, sampling_rate=fs,method="rodrigues2021")

# =====================================================
"plot the detected beats on the original signal to verify that all the beat were correctly detected"
# =====================================================

xx = r_peaks[1]
rpeaks = xx["ECG_R_Peaks"]
WW = signal_input
signal_input = [x+4.7 for x in WW]
pl = [4.700] * len(rpeaks)

fig, axs = plt.subplots()
axs.plot(signal_input, label="the input signal")
axs.plot(rpeaks, pl, 'ro')
axs.set_ylabel('Amplitude')

axs.legend()
plt.show()

# =====================================================
"in this part we change the misdetected peak by the correct position"
# =====================================================

# == extracting the wrong/early detected beat before and after the seizure
# to_change = [1538 ]
# change_to = [1705493, 1706595]
#
# for i in range (len(to_change)):
#     print("i equals to\t",i)
#     print(to_change[i])
#     search = to_change[i]
#     sei = [i for i, e in enumerate(rpeaks) if e == search]
#     print(sei)
#     sei = int(sei[0])
#     rpeaks[sei] = change_to[i]
# # # # #

# =====================================================
"In this part, if the there's a false detected beats, we delete them"
# =====================================================

# to_del = [161489, 409505, 541515, 54258, 5732047, 652432, 725635, 735843]
# peak = rpeaks.tolist()
# print("length of r_peaks\t", len(peak))
# print("type of r_peaks\t", type(peak))
# for i in range (len(to_del)):
#     search = to_del[i]
#     print('searshing for this value:\t',search)
#     sei = [i for i, e in enumerate(peak) if e == search]
#     sei = int(sei[0])
#     del peak[sei]
# rpeaks = peak

# =====================================================
"Here to add a beats that were not benn detected "
# =====================================================

# add = [366336, 365878, 355897, 652312]
# for i in range(len(add)):
#     rpeaks.append(add[i])
# rpeaks.sort()


# =====================================================
" saving all the results"
# =====================================================
path = ""
Path(path).mkdir(parents=True, exist_ok=True)

# np.savetxt(path + 'peaks-'+ ID + '.txt', np.array(rpeaks))
# np.savetxt(path + 'signal-'+ ID + '.txt', np.array(signal_input))

print("end of your code ")