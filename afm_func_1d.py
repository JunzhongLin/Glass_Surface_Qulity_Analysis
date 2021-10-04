import numpy as np
import heapq
from scipy.signal import find_peaks, savgol_filter
from matplotlib import pyplot as plt
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Maximum height (Rz) Represents the sum of the maximum peak height Zp and the maximum valley depth Zv of a profile
# within the reference length. Indicated as Ry within JIS’94 Profile peak: Portion above (from the object) the mean
# profile line (X-axis) Profile valley: Portion below (from the surrounding space) the mean profile line (X-axis) Pz
# Maximum height of the primary profile Wz  Maximum height of the waviness
def Rz(input_data, **kwargs):
    Rp = np.max(input_data, axis=1)
    Rv = np.abs(np.min(input_data, axis=1))
    Rz = Rp + Rv
    # print("The maximum height Rz is:", Rz)
    return Rz


# Maximum profile peak height (Rp)
# Represents the maximum peak height Zp of a profile within the sampling length.
# Pp  The maximum peak height of the primary profile
def Rp(input_data, **kwargs):
    Rp = np.max(input_data, axis=1)
    return Rp


# Maximum profile valley depth (Rv)
# Represents the maximum peak height Zp of a profile within the sampling length.
# Pv　The maximum peak height of the primary profile
# Wv　The maximum peak height of the waviness profile
def Rv(input_data, **kwargs):
    Rv = np.abs(np.min(input_data, axis=1))
    return Rv


# Mean height (Rc)
# Represents the mean for the height Zt of profile elements within the sampling length.
#
# Profile element: A set of adjacent peaks and valleys
# Minimum height and minimum length to be discriminated from the peaks (valleys)
# Minimum height discrimination: 10% of the Rz value
# Minimum length discrimination: 1% of the reference length
def func_Rc(input_data): ## need to be revised
    Rc = (np.sum(abs(input_data))) / input_data.shape[1]
    print("The Mean Height Rc is:  ", Rc)
    return Rc


# Total height (Rt) Represents the sum of the maximum peak height Zp and the maximum valley depth Zv of a profile
# within the evaluation length, not sampling length.
# Relationship Rt≧Rz applies for all profiles
# Pt  The maximum total height of the profile (Max in the case of JIS’82)
# Wt  The maximum total height of the waviness
# Note Rt is a stricter standard than Rz in that the measurement is conducted against the evaluation length. It
# should be noted that the parameter is significantly influenced by scratches, contamination, and measurement noise
# due to its utilization of peak values.

def Rt(input_data, **kwargs):
    Zpi = np.max(input_data, axis=1)
    Zvi = np.abs(np.min(input_data, axis=1))
    Rt = Zpi + Zvi
    return Rt


# Ten-point mean roughness (Rzjis) Represents the sum of the mean value for the height of the five tallest peaks and
# the mean of the depth of the five deepest valleys of a profile within the sampling length.
# Indicated as Rz within JIS’94 Note Rzjis is equivalent to the parameter Rz of the obsolete JIS standard B0601:1994.
# Although ten-point mean roughness was deleted from current ISO standards, it was popularly used in Japan and was
# retained within the JIS standard as the parameter Rzjis.


def Rzjis(input_data, use_smooth=True, window_length=31, polyorder=5, search_range=10, **kwargs):

    if use_smooth:  # the prior modulus provided the smooth data
        all_data = input_data
    else:           # the prior modulus provided the raw data
        all_data = savgol_filter(input_data, window_length=window_length,
                                 polyorder=polyorder, axis=-1)

    peaks_pos = []
    valleys_pos = []
    size_y = input_data.shape[1]
    res = np.zeros(len(input_data))

    # plot figure
    # fig, ax = plt.subplots(4, 4)

    for i, data in enumerate(all_data):
        peak_pos, _ = find_peaks(data, prominence=data.max() * 0.1)
        valley_pos, _ = find_peaks(-data, prominence=-data.min() * 0.1)
        peaks_pos.append(peak_pos)
        valleys_pos.append(valley_pos)
        if use_smooth:
            peaks = data[peak_pos]
            valleys = data[valley_pos]
            num = min(len(peak_pos), len(valley_pos)) if min(len(peak_pos), len(valley_pos)) < 5 else 5
            res[i] = (np.sum(np.abs(np.sort(peaks)[-num:])) + np.sum(np.abs(np.sort(valleys)[:num]))) / num
            ## figure plot for debug
            # ax[i // 4, i % 4].plot(all_data[i], c='g')
            # pos_for_plot = peak_pos[np.where(peaks >= np.sort(peaks)[-num])[0]]
            # value_for_plot = peaks[np.where(peaks >= np.sort(peaks)[-num])[0]]
            # ax[i // 4, i % 4].scatter(pos_for_plot, value_for_plot, c='r', marker='o')
            # pos_for_plot = valley_pos[np.where(valleys <= np.sort(valleys)[num - 1])[0]]
            # value_for_plot = valleys[np.where(valleys <= np.sort(valleys)[num - 1])[0]]
            # ax[i // 4, i % 4].scatter(pos_for_plot, value_for_plot, c='r', marker='x')
        else:
            peak_raw = []
            valley_raw = []
            for pos in peak_pos:
                peak_raw.append(
                    np.max(input_data[i, :][
                           0 if int(pos - search_range / 2) else int(pos - search_range / 2):
                           size_y - 1 if int(pos + search_range / 2) > size_y - 1 else int(pos + search_range / 2)])
                )
            for pos in valley_pos:
                valley_raw.append(
                    np.min(input_data[i, :][
                           0 if int(pos - search_range / 2) else int(pos - search_range / 2):
                           size_y - 1 if int(pos + search_range / 2) > size_y - 1 else int(pos + search_range / 2)])
                )
            num = min(len(peak_raw), len(valley_raw)) if min(len(peak_raw), len(valley_raw)) < 5 else 5
            res[i] = (np.sum(np.abs(np.sort(peak_raw)[-num:])) + np.sum(np.abs(np.sort(valley_raw)[:num]))) / num
    return res
    # locate the peak position:



# Arithmetic mean deviation (Ra)
# Represents the arithmetric mean of the absolute ordinate Z(x) within the sampling length.
#
# Pa  The arithmetic mean height of the primary profile
# Wa  The arithmetic mean waviness
#
# Note One of the most widely used parameters is the mean of the average height difference for the average surface.
# It provides for stable results as the parameter is not significantly influenced by scratches, contamination,
# and measurement noise.
def Ra(input_data, **kwargs):
    Zx = np.abs(input_data)
    Ra = np.sum(Zx, axis=1) / input_data.shape[1]
    # print("The Arithmetic mean deviation (Ra) is:", Ra)
    return Ra


# Root mean square deviation (Rq)
# Represents the root mean square for Z(x) within the sampling length.
def Rq(input_data, **kwargs):
    Zx = np.abs(input_data)
    Rq = np.sqrt(np.sum(np.square(Zx), axis=1) / input_data.shape[1])
    # print("The Root mean square deviation (Rq) is:  ", Rq)
    return Rq


# Skewness (Rsk)
# The quotient of the mean cube value of Z (x) and the cube of R8 within a sampling length.
#
# Rsk=0: Symmetric against the mean line (normal distribution)
# Rsk>0: Deviation beneath the mean line
# Rsk<0: Deviation above the mean line
def Rsk(input_data, **kwargs):
    temp_Rq = Rq(input_data)
    Rsk = (np.sum(np.power(input_data, 3), axis=1) / input_data.shape[1]) / np.power(temp_Rq, 3)
    # print("The Skewness (Rsk) is:  ", Rsk)
    return Rsk


# Kurtosis (Rku)
# The quotient of the mean quadratic value of Z (x) and the fourth power of Rq within a sampling length.
#
# Rku=3: Normal distribution
# Rku>3: The height distribution is sharp
# Rku<3: The height distribution is even
def Rku(input_data, **kwargs):
    temp_Rq = Rq(input_data)
    Rku = (np.sum(np.power(input_data, 4), axis=1) / input_data.shape[1]) / np.power(temp_Rq, 4)
    # print("The Kurtosis (Rku) is:  ", Rku)
    return Rku


# Mean width (RSm)
# Represents the mean for the length Xs of profile elements within the sampling length.
#
# Indicated as Sm within JIS’94
# Minimum height and minimum length to be discriminated from peaks (valleys)
# Minimum height discrimination: 10% of the Rz value
# Minimum length discrimination: 1% of the reference length
def func_Rsm(input_data):
    pass


# Root mean square slope (Rdq)
# Represents the root mean square for the local slope dz/dx within the sampling length.
def func_Rdq(input_data):
    diff = np.sum(np.diff(input_data))
    Rdq = np.sqrt(np.power(2,diff)/input_data.shape[1])
    print("The Root mean square slope (Rdq) is:  ", Rdq)
    pass


def func_Rmrc(input_data, x=0.05):
    a = np.max(input_data, axis=1).reshape(16, 1)
    C = a * x
    heightc = a - C
    newdata = input_data - heightc
    zeronegative = np.maximum(newdata, 0)
    wewant = np.count_nonzero(zeronegative, axis=1)
    Rmrc = wewant.reshape(16, 1) / input_data.shape[1]
    return Rmrc




def func_Rdc(input_data,Rmrc1,Rmrc2):
    a = np.sort(input_data, axis=1)
    index1 = -np.around(Rmrc1*256).astype('int64')
    index2 = -np.around(Rmrc2*256).astype('int64')
    C1=[]
    C2=[]
    for m in range(0, input_data.shape[0]):
        C1.append(a[m, :][index1[m]])
        C2.append(a[m, :][index2[m]])
    Rdc = (np.array(C1)-np.array(C2)).reshape(1,16)[0, :]
    return Rdc


def func_Rmr(input_data,Rmr0,Rdeltac):
    a = np.sort(input_data,axis=1)
    index = np.around(-Rmr0*256).astype('int64')
    C0 = []
    for n in range(0,input_data.shape[0]):
        C0.append(a[n,:][index[n]])
    C1 = np.array(C0).reshape(16,1) - Rdeltac.reshape(16,1)
    removenegative = np.maximum(input_data-C1,0)
    Rmr = np.count_nonzero(removenegative,axis=1)/input_data.shape[1]
    return Rmr


def K_alpha_ANAM(input_data, tau, alpha, **kwargs):
    size = input_data.shape[1]
    data = input_data   #
    total_ijl = np.zeros(input_data.shape[0])

    for i in range(tau, size-tau):
        total_jl = np.zeros(input_data.shape[0])
        for j in range(0, tau+1):
            for l in range(0, tau+1):
                total_jl += np.power(np.abs(data[:, i+j]-data[:, i-l]), alpha)
        total_ijl += np.power(total_jl, 1/alpha)

    res = (tau+1)**(-2/alpha)/(size-2*tau)*total_ijl

    return res


def linear_func(x, a, b):
    return a*x+b


def delta_ANAM(input_data, tau_list=np.arange(1, 10), alpha=2, **kwargs):
    K_alpha = np.ones([input_data.shape[0], len(tau_list)])
    tau_matrix = np.ones([input_data.shape[0], len(tau_list)])
    for i, tau in enumerate(tau_list):
        tau_matrix[:, i] *= tau
        K_alpha[:, i] = K_alpha_ANAM(input_data, tau=tau, alpha=alpha)

    log_K_alpha = np.log10(K_alpha)
    log_tau_matrix = np.log10(tau_matrix)
    delta_anam = np.zeros(input_data.shape[0])

    for i in range(input_data.shape[0]):
        popt, pcov = curve_fit(linear_func, log_tau_matrix[i, :], log_K_alpha[i, :])
        delta_anam[i] = 2-popt[0]

    return -delta_anam

# Material ratio curve and probability density curves Material ratio curves signify the ratio of materiality derived
# as a mathematical function of parameter c, where c represents the height of severance for a specific sample. This
# is also referred to as the bearing curve (BAC) or Abbott curve. Probability density curves signify the probability
# of occurrence for height Zx. The parameter is equivalent to the height distribution histogram.


# Samplelength and data should be input by users
# Rt should use evaluation length instead of sample length, plz do it by yourself

if __name__ == '__main__':
    with open('./example.pkl', 'rb') as f:
        input_data = pickle.load(f)
    kwargs = {'use_smooth': True, 'alpha': 2, 'tau_list': np.arange(1, 10)}
    res = delta_ANAM(input_data, **kwargs)
