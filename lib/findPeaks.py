"""
-----------------------------------------------------------------------
findPeaks.py

Author: Wisam Reid
-----------------------------------------------------------------------
"""

# Thanks Julius!

from abbreviations import *

def findpeaks(Xwdb, fs, N):

    peaks = []
    freqs = []
    
    length = size(Xwdb)

    # find peaks and order from max amplitude to min
    for sample in range(1,length-1):

        if (abs(Xwdb[sample]) > abs(Xwdb[sample-1]) and abs(Xwdb[sample]) > abs(Xwdb[sample+1]) and 10.0* log10(abs(Xwdb[sample]))>-30.0):

            peaks = np.append(peaks,Xwdb[sample])
            freqs = np.append(freqs,sample)

    peaks = peaks.astype(int)
    freqsIndex = freqs.astype(int)

    # parabolic interpolation
    estimateFreqs = []
    estimateAmp = []

    for idx in range(0,len(freqs)):

        a = abs(Xwdb[freqs[idx]-1])
        b = abs(Xwdb[freqs[idx]])
        r = abs(Xwdb[freqs[idx]+1])
        p = (1/2)*(a-r)/(a+r-2*b)
        A = b-(a-r)*(p/4)
        estimateFreqs = append(estimateFreqs,(freqs[idx]+p)*(fs/N))
        estimateAmp = append(estimateAmp,A)

    return estimateAmp, estimateFreqs, freqsIndex


