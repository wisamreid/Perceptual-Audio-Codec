import numpy as np
from psychoac import *

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    BITS PER LINE IN EACH BAND
    """
    bitsPerLine=np.zeros(len(nLines))
    bitsLeft=bitBudget
    counter=0

    while bitsLeft-nLines[counter]>0:
        bitsPerLine[counter]+=1
        bitsLeft-=nLines[counter]
        counter=(counter+1)%len(nLines)

    bitsPerLine[bitsPerLine==1]=0
    bitsPerLine=bitsPerLine.astype('int')

    # print sum(bitsPerLine*nLines)
    return bitsPerLine

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    peaks=peakSPL.copy()
    bitsLeft=bitBudget
    bitsPerLine=np.zeros(len(nLines))

    while bitsLeft>0:
        if sum(bitsPerLine)>=maxMantBits*len(nLines):
            break
        big=np.argmax(peaks)
        if bitsLeft-nLines[big]<0:
            break
        peaks[big]-=6.
        if bitsPerLine[big]>=maxMantBits:
            continue
        bitsPerLine[big]+=1
        bitsLeft-=nLines[big]

    bitsPerLine[bitsPerLine==1]=0
    bitsPerLine=bitsPerLine.astype('int')

        # The solution allocates the last bits to the highest peaks that will fit

    # print sum(bitsPerLine*nLines)
    return bitsPerLine

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """

    peaks=SMR.copy()
    bitsLeft=bitBudget
    bitsPerLine=np.zeros(len(nLines))

    while bitsLeft>0:
        if sum(bitsPerLine)>=maxMantBits*len(nLines):
            break
        big=np.argmax(peaks)
        if bitsLeft-nLines[big]<0:
            break
        peaks[big]-=6.
        if bitsPerLine[big]>=maxMantBits:
            continue
        bitsPerLine[big]+=1
        bitsLeft-=nLines[big]

        # The solution allocates the last bits to the highest peaks that will fit

    bitsPerLine[bitsPerLine==1]=0
    bitsPerLine=bitsPerLine.astype('int')

    # print sum(bitsPerLine*nLines)
    return bitsPerLine

# Question 1.c)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Return:
            bits[nBands] is number of bits allocated to each scale factor band

        Logic:
           Maximizing SMR over block gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).
    """
    peaks=SMR.copy()
    bitsLeft=bitBudget
    bitsPerLine=np.zeros(len(nLines))

    while bitsLeft>0:
        if sum(bitsPerLine)>=maxMantBits*len(nLines):
            break
        big=np.argmax(peaks)
        if bitsLeft-nLines[big]<0:
            break
        peaks[big]-=6.
        if bitsPerLine[big]>=maxMantBits:
            continue
        bitsPerLine[big]+=1
        bitsLeft-=nLines[big]

        # The solution allocates the last bits to the highest peaks that will fit

    bitsPerLine[bitsPerLine==1]=0
    bitsPerLine=bitsPerLine.astype('int')
    
    # print sum(bitsPerLine*nLines)
    return bitsPerLine

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    # import bitalloc_ as bitsoln
    # print bitsoln.BitAlloc.__doc__

    execfile("HW5.py")