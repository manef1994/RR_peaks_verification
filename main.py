import bisect
import numpy as np
from ecgdetectors import Detectors
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


# path = "E:\\data\\Fabrice\\Patients\\PN09\\"
# ID = "signal-PN02-220310E-CEX_0008"
# # path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\Interictal\\PN05\\"
# # ID = "PN05-2-10_40"
# fs = 256
# # # == load the epileptic patient
# with open(path + ID + ".txt", 'r') as file1:
#     signal_input = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

# == reading the EDf file and extracting the ECG signal
# file_path = "C:\\Users\\Ftay\\Desktop\PhD\\tests\\data\\siena-scalp-eeg-database-1.0.0\\PN06\\PN06-4.edf"
path = "E:\\data\\Fabrice\\Patients\\PN02\\"
ID = "220310E-CEX_0008"
file_path = path + ID + ".edf"
edf = read_raw_edf(file_path, preload=False, stim_channel=None, verbose=False)
xx = edf.ch_names
index = xx.index("ECG")
fs = edf.info['sfreq']
fs = int(fs)
signal_input = edf[index]
signal = signal_input[0]

signal_input = signal[0]
print("asba")

# 16:13.23
# end = (fs * 60 * 60 * 1) + (fs * 60 * 48) + (59 * fs)
# seizure = (fs * 60 * 60 * 0) + (fs * 60 * 53) + (07 * fs)
# pre_ictal = (fs * 60 * 60 * 0) + (fs * 60 * 38) + (59 * fs)
#
# print('pre-ictal:\t', pre_ictal)
# print('seizure:\t', seizure)
# print('end:\t', end)

# signal_input = signal_input[pre_ictal:end]
# signal_input = signal_input[pre_ictal:len(signal_input)]

# signal_input_1 = signal_input[104200:len(signal_input)]
# signal_input = signal_input_1
# #
# # == delet unwanted parts of the signals
# signal_input_1 = signal_input[0:364250]
# signal_input1 = signal_input[366300:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# signal_input_1 = signal_input[0:374200]
# signal_input1 = signal_input[380800:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # # # # # # # # # # # # # # # # #
# signal_input_1 = signal_input[0:663500]
# signal_input1 = signal_input[667200:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # # # # # # # # # # # # # # # #
# signal_input_1 = signal_input[0:857000]
# signal_input1 = signal_input[860400:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # # # # # # # # # # # #
# signal_input_1 = signal_input[0:907400]
# signal_input1 = signal_input[910650:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # # # # #
# signal_input_1 = signal_input[0:941500]
# signal_input1 = signal_input[947500:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # # # # # #
# signal_input_1 = signal_input[0:1258100]
# signal_input1 = signal_input[1259400:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # # #
# signal_input_1 = signal_input[0:1482200]
# signal_input1 = signal_input[1485000:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # # #
# signal_input_1 = signal_input[0:647100]
# signal_input1 = signal_input[65200:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # # #
# signal_input_1 = signal_input[0:1229000]
# signal_input1 = signal_input[1234000:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:494500]
# signal_input1 = signal_input[502200:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:533500]
# signal_input1 = signal_input[536400:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:543900]
# signal_input1 = signal_input[545700:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:562200]
# signal_input1 = signal_input[578700:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:587750]
# signal_input1 = signal_input[592500:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:613950]
# signal_input1 = signal_input[620800:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:622800]
# signal_input1 = signal_input[633150:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:625350]
# signal_input1 = signal_input[648600:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:628500]
# signal_input1 = signal_input[631400:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:636775]
# signal_input1 = signal_input[637750:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:638700]
# signal_input1 = signal_input[640600:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:639990]
# signal_input1 = signal_input[648950:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:667100]
# signal_input1 = signal_input[673900:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:662200]
# signal_input1 = signal_input[667200:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1
# # # #
# signal_input_1 = signal_input[0:681200]
# signal_input1 = signal_input[682400:len(signal_input)]
# signal_input_1 = np.append(signal_input_1, signal_input1)
# signal_input = signal_input_1

signal_output = filter_signal(data=signal_input, cutoff=60, sample_rate=fs, order=2, filtertype="notch")
signal_output = filter_signal(data=signal_output, cutoff=20, sample_rate=fs, order=2, filtertype="highpass")
signal_output = filter_signal(data=signal_output, cutoff=10, sample_rate=fs, order=2, filtertype="lowpass")
spectral_welch = []
Samp_entropy, fuzzy_entropy, shannon_entropy, approximate_entropy, approximate_entropy2, multi_multiscale, approximate_entropy3, approximate_entropy4, approximate_entropy5, spectral = (
[] for i in range(10))

six_seconds = fs * 6
one_minut = fs * 60
###########################################################
# == compute the HRV
five_min = []
HRV = []
RRi = []
final_peaks = []

print('fs equals to: ', fs)

RRi = []
spectrale_welch = []
condition = True

first = 10
tt = [first]

start = 0
end = fs * 120

print('signal duration\t\t', ((len(signal_input) / fs) / 60))
print('start to compute the entropy features:')
HRV = []

r_peaks = nk.ecg_peaks(signal_input, sampling_rate=fs,method="rodrigues2021")

xx = r_peaks[1]
rpeaks = xx["ECG_R_Peaks"]

WW = signal_input
signal_input = [x+4.7 for x in WW]

pl = [4.6992] * len(rpeaks)

fig, axs = plt.subplots()
axs.plot(signal_input, label="the input signal")
# axs.plot(rpeaks, signal_input[rpeaks], 'ro')
# axs.axvline((fs * 60 * 40) + (49 * fs), color='r', linestyle='--')
axs.axvline( len(signal_input) - 38400, color='red', linestyle='--')
axs.plot(rpeaks, pl, 'ro')
axs.set_ylabel('Amplitude')

axs.legend()
plt.show()

# ####################################################################

# == computing the RRi to plot it
# detectors = Detectors(fs)
# peaks = detectors.two_average_detector(signal_output)

# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\unhealthy\\"
# ID = "PN00-4"
# fs = 512
# # # == load the epileptic patient
# with open(path + "peaks-" + ID + ".txt", 'r') as file1:
#     peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]


# print("first peak\t", peaks[0])
# print("final peak\t", peaks[-1])
#peaks = peaks - 11
# peaks = [x - 38 for x in peaks]

# r_peaks = nk.ecg_peaks(signal_input, sampling_rate=fs,method="rodrigues2021")
#
# xx = r_peaks[1]
# peaks = xx["ECG_R_Peaks"]
#
# pl = [0] * len(peaks)
#
# print("lenght of peaks:\t", len(peaks))
# # == fixing the late detection of Peak RR
# #peaks = [x - 8 for x in peaks]
#
# # == plot the orifinal signal with the beat detected
# fig, axs = plt.subplots()
# axs.plot(signal_input, label="the input signal")
# # axs.plot(peaks, signal_input[peaks], 'ro')
# axs.plot(peaks, pl, 'ro')
# axs.set_ylabel('Amplitude')
# axs.axvline((fs * 60 * 53) + (7 * fs), color='red', linestyle='--')
# #axs.axvline(x = seizure, color='red', linestyle='--')
# # axs.grid(True)
# axs.legend()
# plt.show()
# print('seizure index')
# print ('stop')

# == extracting the wrong/early detected beat before and after the seizure
# to_change = [1538 ]
# change_to = [1705493, 1706595]
#
# for i in range (len(to_change)):
#     print("i equals to\t",i)
#     print(to_change[i])
#     search = to_change[i]
#     sei = [i for i, e in enumerate(peaks) if e == search]
#     print(sei)
#     sei = int(sei[0])
#     peaks[sei] = change_to[i]
# # # # #
# # #
# to_del = [198112, 386784, 387008, 415703, 602818, 644816]
#
# peak = rpeaks.tolist()
# rpeaks = peak
# print("length of r_peaks\t", len(peak))
# print("type of r_peaks\t", type(peak))
# for i in range (len(to_del)):
#     search = to_del[i]
#     print('searshing for this value:\t',search)
#     sei = [i for i, e in enumerate(peak) if e == search]
#     sei = int(sei[0])
#     del peak[sei]
# # #
# # #
# add = [1706309, 1711376]
# for i in range(len(add)):
#     peaks.append(add[i])
# peaks.sort()
# ####################################################################
# # == saving all the results
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\Peaks_RR\\fabrice\\"
# # path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\PN013"
# Path(path).mkdir(parents=True, exist_ok=True)
#
# np.savetxt(path + '\\peaks-.txt', np.array(peaks))

# # == saving all the results
path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\Peaks_RR\\fabrice\\"
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\PN013"
Path(path).mkdir(parents=True, exist_ok=True)

np.savetxt(path + '\\peaks-'+ ID + '.txt', np.array(rpeaks))

print("end of your code ")

