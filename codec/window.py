"""
-----------------------------------------------------------------------
window.py -- Defines functions to window an array of data samples

Author: Wisam Reid
-----------------------------------------------------------------------
"""

# from mdct_sol import *
# from window_sol import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
# import math
# from numpy.fft import fft, ifft
# from abbreviations import *
from mdct import *


# Booleans
problem1d = False
problem1e = True

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    N = float(len(dataSampleArray))
    t = arange(N)

    dataSampleArray *= sin((t + 0.5) * pi / N)

    return dataSampleArray

def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    N = float(len(dataSampleArray))
    t = arange(N)

    dataSampleArray *= 0.5 * (1 - cos(2.0 * (t + 0.5) * pi / N))

    return dataSampleArray

### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following pp. 108-109 and pp. 117-118 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    N = float(len(dataSampleArray))
    t = arange(int(N/2))

    # i0 --> 0th modified bessel function
    kaiser = cumsum(i0(alpha * pi * sqrt(1.0 - (4.0 * t / N - 1.0) ** 2)))

    # including boundary value
    denominator = kaiser[-1] + i0(alpha * pi * sqrt(1.0 - (4.0 * N / 2.0 / N - 1.0) ** 2))

    window = sqrt(kaiser / denominator)
    dataSampleArray *= concatenate((window, window[::-1]), axis=0)

    return dataSampleArray


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    if problem1d: # windows

        N = 1024
        n = arange(N)

        # figure and plots
        figure(figsize=(10, 7))
        plot(n, SineWindow(ones(N)), 'g', label='Sine Window')
        plot(n, HanningWindow(ones(N)), 'b', label='Hanning Window')
        plot(n, KBDWindow(ones(N)), 'r', label='KBD Window with $\\alpha=4$')

        # labels
        xlabel('n (Samples)')
        xlim(0, N - 1)
        ylabel('Amplitude')
        title('Implemented Windows')
        legend(loc=8)

        # display
        grid()
        show()

    if problem1e: # resolution

        FS = 44100.0
        N = 1024
        n = arange(N)
        freq = 3000.0

        # signal
        x = cos(2 * pi * freq * n / FS)

        # windowed signals
        x_Sine = SineWindow(x)
        x_Hann = HanningWindow(x)
        x_KBD = KBDWindow(x)

        # KBD window gain factor
        w_KBD = 1.0 / N * sum(KBDWindow(ones(N)))



        ###### FFT ######

        # FFT Sine
        fft_x_Sine = fft(x_Sine)[:N / 2]
        SPL_fft_x_Sine = 96.0 + 10.0 * log10(8.0 / N ** 2 * abs(fft_x_Sine) ** 2)

        # FFT Hann
        fft_x_Hann = fft(x_Hann)[:N / 2]
        SPL_fft_x_Hann = 96.0 + 10.0 * log10(8.0 / 3.0 * 4.0 / N ** 2 * abs(fft_x_Hann) ** 2)

        # FFT KBD
        fft_x_KBD = fft(x_KBD)[:N / 2]
        SPL_fft_x_KBD = 96.0 + 10.0 * np.log10(1.0 / w_KBD * 4.0 / N ** 2 * abs(fft_x_KBD) ** 2)



        ###### MDCT ######

        # MDCT Sine
        mdct_x_Sine = MDCT(x_Sine, N / 2, N / 2)
        SPL_mdct_x_Sine = 96.0 + 10.0 * log10(4.0 * mdct_x_Sine ** 2)

        # MDCT Hann
        mdct_x_Hann = MDCT(x_Hann, N / 2, N / 2)
        SPL_mdct_x_Hann = 96.0 + 10.0 * log10(4.0 * mdct_x_Hann ** 2)

        # MDCT KBD
        mdct_x_KBD = MDCT(x_KBD, N / 2, N / 2)
        SPL_mdct_x_KBD = 96.0 + 10.0 * log10(4.0 / w_KBD * mdct_x_KBD ** 2)

        freq_fft = arange(N / 2) * FS / N
        freq_mdct = freq_fft + FS / N / 2.0

        figure()
        plot(freq_fft, SPL_fft_x_Sine, 'r-', freq_fft, SPL_fft_x_Hann, 'g-', freq_fft, SPL_fft_x_KBD, 'b-', freq_mdct, SPL_mdct_x_Sine, 'k-', freq_mdct, SPL_mdct_x_Hann, 'm-', freq_mdct, SPL_mdct_x_KBD, 'c-')
        ylim(-100, 100)
        xlim(0, FS / 2)

        title('FFT vs MDCT')
        xlabel('Freq (Hz)')
        ylabel('SPL (dB)')

        grid()
        plt.legend(('FFT: Sine Window','FFT: Hanning Window','FFT: KBD Window', 'MDCT: Sine Window','MDCT: Hanning Window', 'MDCT: KBD Window with $\\alpha=4$'))
        plt.grid()

        figure()
        plot(freq_fft, SPL_fft_x_Sine, 'r-', freq_fft, SPL_fft_x_Hann, 'g-', freq_fft, SPL_fft_x_KBD, 'b-', freq_mdct, SPL_mdct_x_Sine, 'k-', freq_mdct, SPL_mdct_x_Hann, 'm-', freq_mdct, SPL_mdct_x_KBD, 'c-')
        ylim(0, 100)
        xlim(freq - 1000, freq + 1000)

        xlabel('Freq (Hz)')
        ylabel('SPL (dB)')
        title('Resolution Comparison')
        grid()
        show()
