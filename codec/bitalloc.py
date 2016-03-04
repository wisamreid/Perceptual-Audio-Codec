# from inspect import getsourcefile
# import os.path as path, sys
# current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
# sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
# from lib.abbreviations import *
from window import *
from mdct import *
from psychoac import *
# import provided.bitalloc_sol as solution

infinity = float('inf')
negative_infinity = float('-inf')

# Test_Booleans = [True, True, True, True]

# test_uniform_bitalloc = Test_Booleans[0]
# test_constant_SNR_bitalloc = Test_Booleans[1]
# test_constant_MNR_bitalloc = Test_Booleans[2]
# test_bitalloc = Test_Booleans[3]

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """

    allocation = ones(nBands, dtype=int)
    bits_per_line = int(bitBudget / float(sum(nLines)))

    allocation = allocation * bits_per_line

    # distribute left over bits
    remaining_bits = bitBudget - (sum(allocation*nLines))

    if remaining_bits:

        line = 0
        while remaining_bits > 0:

            remaining_bits -= nLines[line%nBands]

            if(remaining_bits < 0):
                break

            if (allocation[line%nBands] < maxMantBits):
                allocation[line%nBands] += 1

            line += 1

    # zero out 1 bit allocations (Mid-Tread)
    allocation[allocation < 2] = 0
    # no overflow
    allocation[allocation > maxMantBits] = maxMantBits

    return allocation


def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """

    allocation = zeros(nBands, dtype=int)
    remaining_bits = bitBudget
    noise_floor = peakSPL * ones(nBands)
    dB_per_bit = 6.0

    while remaining_bits > 0:

        max_smr_band = noise_floor.argmax()

        if allocation[max_smr_band] < maxMantBits and (remaining_bits - nLines[max_smr_band]) >= 0:

            allocation[max_smr_band] += 1
            remaining_bits -= nLines[max_smr_band]

        noise_floor[max_smr_band] -= dB_per_bit

    # zero out 1 bit allocations (Mid-Tread)
    allocation[allocation < 2] = 0
    # no overflow
    allocation[allocation > maxMantBits] = maxMantBits

    return allocation


def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """

    allocation = zeros_like(nLines, dtype=int)
    remaining_bits = bitBudget
    dB_per_bit = 6.0
    noise_floor = SMR.copy()

    while remaining_bits > 0:

        max_smr_band = argmax(noise_floor)

        if allocation[max_smr_band] < maxMantBits and (remaining_bits - nLines[max_smr_band]) >= 0:

            allocation[max_smr_band] += 1
            remaining_bits -= nLines[max_smr_band]

        noise_floor[max_smr_band] -= dB_per_bit


    # zero out 1 bit allocations (Mid-Tread)
    allocation[allocation < 2] = 0
    # no overflow
    allocation[allocation > maxMantBits] = maxMantBits

    return allocation


# Question 1.c)
def BitAlloc(bitBudget, extraBits, maxMantBits, nBands, nLines, SMR, LRMS):
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
    MS_Threshold = -5.
    LR_Threshold = -15.

    while valid.any():
        iMax = np.arange(nBands)[valid][np.argmax((SMR-bits*6.)[valid])]
        if LRMS[iMax]: # MS
            if max(SMR-(bits-1)*6.)<(-5.0): valid[iMax] = False
        else: # LR
            if max(SMR-(bits-1)*6.)<(-15.0): valid[iMax] = False
        # print max(SMR-(bits-1)*6.)
        if (totalBits - nLines[iMax]) >=0:
            bits[iMax] += 1
            totalBits -= nLines[iMax]
            if bits[iMax] >= maxMantBits:
                valid[iMax] = False
        else:
            valid[iMax] = False

    if max(SMR-(bits-1)*6.)<((MS_Threshold+LR_Threshold)/2.): print '*'
    totalBits+=sum(nLines[bits==1])
    bits[bits==1]=0

    bitDifference=totalBits-extraBits

    return bits,bitDifference


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":


    FS = 48000.0
    N = 1024

    # allocation variables
    maxMantBits = 16
    scale = 4

    # construct input signal
    freqs = (420, 530, 640, 840, 4200, 8400)
    amps = (0.60, 0.11, 0.10, 0.08, 0.05, 0.03)
    n = arange(N, dtype=float)
    x = zeros_like(n)

    for i in range(len(freqs)):
        x += amps[i] * cos(2 * pi * freqs[i] * n / FS)

    nLines = AssignMDCTLinesFromFreqLimits(N/2, FS)
    sfb = ScaleFactorBands(nLines)
    nBands = sfb.nBands
    nLines = sfb.nLines

    bitBudget = int(128.0/(FS/1000.0) * N/2 - (4+4) * nBands - 4)

    mdct = (1 << scale) * MDCT(SineWindow(x), N/2, N/2)
    SMR = CalcSMRs(x, mdct, scale, FS, sfb)


    if test_uniform_bitalloc:

        print "BitAllocUniform Test Results:"
        print "Solution:"
        print solution.BitAllocUniform(bitBudget,maxMantBits,nBands,nLines)
        print "Generated:"
        print BitAllocUniform(bitBudget,maxMantBits,nBands,nLines)


    if test_constant_SNR_bitalloc:

        mdct_spl = SPL(4.0 * (mdct**2.0))
        peaks = zeros(sfb.nBands)

        for band in range(nBands):

            max = mdct_spl[sfb.lowerLine[band]]

            for line in range(sfb.lowerLine[band]+1,sfb.upperLine[band]+1):

                SPL = mdct_spl[line]

                if (SPL > max):
                    max = SPL

            peaks[band] = max

        print "BitAllocConstSNR Test Results:"
        print "Solution:"
        print solution.BitAllocConstSNR(bitBudget,maxMantBits,nBands,nLines,peaks).astype(int)
        print "Generated:"
        print BitAllocConstSNR(bitBudget,maxMantBits,nBands,nLines,peaks)


    if test_constant_MNR_bitalloc:

        print "BitAllocConstMNR Test Results:"
        print "Solution:"
        print solution.BitAllocConstMNR(bitBudget,maxMantBits,nBands,nLines,SMR).astype(int)
        print "Generated:"
        print BitAllocConstMNR(bitBudget,maxMantBits,nBands,nLines,SMR)


    if test_bitalloc:

        print "BitAlloc Test Results:"
        print "Solution:"
        print solution.BitAlloc(bitBudget,maxMantBits,nBands,nLines,SMR)
        print "Generated:"
        print BitAlloc(bitBudget,maxMantBits,nBands,nLines,SMR)
