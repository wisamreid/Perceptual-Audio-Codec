from window import *
from mdct import *
from psychoac import *


infinity = float('inf')
negative_infinity = float('-inf')

# Question 1.c)
def BitAlloc(bitBudget, extraBits, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           extraBits is the amount of leftover bits from other blocks
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Return:
            bits[nBands] is number of bits allocated to each scale factor band
            bitDifference is the net gain/loss over the original bit budget

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

    bits = np.zeros(nBands, dtype=int)
    valid = np.ones(nBands, dtype=bool)
    totalBits = int(bitBudget+extraBits)

    while valid.any():
        iMax = np.arange(nBands)[valid][np.argmax((SMR-bits*6.)[valid])]
        if max(SMR-(bits-1)*6.)<(-15.0): valid[iMax] = False
        # print max(SMR-(bits-1)*6.)
        if (totalBits - nLines[iMax]) >=0:
            bits[iMax] += 1
            totalBits -= nLines[iMax]
            if bits[iMax] >= maxMantBits:
                valid[iMax] = False
        else:
            valid[iMax] = False

    if max(SMR-(bits-1)*6.)<(-15.): print '*'
    totalBits+=sum(nLines[bits==1])
    bits[bits==1]=0

    bitDifference=totalBits-extraBits

    return bits,bitDifference


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass
