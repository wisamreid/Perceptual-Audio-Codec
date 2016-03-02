import numpy as np
import bitalloc_ as sol
from psychoac import *

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    totalBitBudget = bitBudget
    bitsEachLineInBand = np.zeros_like(nLines)
    nBands = len(nLines)
    count = 0
    while totalBitBudget > 0:
        index = count%nBands
        totalBitBudget -= nLines[index]
        if(totalBitBudget<0):
            break
        bitsEachLineInBand[index] += 1
        count += 1
    bitsEachLineInBand[bitsEachLineInBand==1] = 0
    return bitsEachLineInBand

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    totalBitBudget = bitBudget
    bitsEachLineInBand = np.zeros_like(nLines)
    myPeakSPL = peakSPL*np.ones(nBands)
    nBands = len(nLines)
    while True:
        if(myPeakSPL==-1e9*np.ones(nBands)).all():
            break
        maxSPLIndex = np.argmax(myPeakSPL)
        if(totalBitBudget>=nLines[maxSPLIndex]):
            myPeakSPL[maxSPLIndex] = myPeakSPL[maxSPLIndex] - 6.0
            bitsEachLineInBand[maxSPLIndex] += 1
            totalBitBudget -= nLines[maxSPLIndex]
            if(bitsEachLineInBand[maxSPLIndex]==maxMantBits):
                myPeakSPL[maxSPLIndex] = -1e9
        else:
            myPeakSPL[maxSPLIndex] = -1e9
    bitsEachLineInBand[bitsEachLineInBand==1] = 0
    return bitsEachLineInBand

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    totalBitBudget = bitBudget
    bitsEachLineInBand = np.zeros_like(nLines)
    mySMR = SMR*np.ones(nBands)
    nBands = len(nLines)
    while True:
        if(mySMR==-1e9*np.ones(nBands)).all():
            break
        maxSMRIndex = np.argmax(mySMR)
        if(totalBitBudget>=nLines[maxSMRIndex]):
            mySMR[maxSMRIndex] = mySMR[maxSMRIndex] - 6.0
            bitsEachLineInBand[maxSMRIndex] += 1
            totalBitBudget -= nLines[maxSMRIndex]
            if(bitsEachLineInBand[maxSMRIndex]==maxMantBits):
                mySMR[maxSMRIndex] = -1e9
        else:
            mySMR[maxSMRIndex] = -1e9
    bitsEachLineInBand[bitsEachLineInBand==1] = 0
    return bitsEachLineInBand

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
           Maximizing SMR over blook gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).
    """
    bitsEachLineInBand = np.zeros_like(nLines)
    avgSMR = sum(nLines*SMR)/sum(nLines)
    for i in range(nBands):
        R = float(bitBudget)/sum(nLines) + 1.0 * (SMR[i] - avgSMR)/6.0
        if R<2:
            R = 0
        if R>maxMantBits:
            R = maxMantBits
        bitsEachLineInBand[i] = int(R)
    ##  Take bits back to meet the budget if overshoot ##
    totalBits = sum(bitsEachLineInBand * nLines)
    mySMR = SMR*np.ones(nBands)
    while True:
        if(mySMR==1e9*np.ones(nBands)).all():
            break
        minSMRIndex = np.argmin(mySMR)
        if(totalBits>=bitBudget):
            mySMR[minSMRIndex] = mySMR[minSMRIndex] + 6.0
            if(bitsEachLineInBand[minSMRIndex]!=0):
                bitsEachLineInBand[minSMRIndex] -= 1
                if (bitsEachLineInBand[minSMRIndex] == 1):
                    bitsEachLineInBand[minSMRIndex] = 0
            totalBits = sum(bitsEachLineInBand * nLines)
            if(bitsEachLineInBand[minSMRIndex]==0):
                mySMR[minSMRIndex] = 1e9
        else:
            mySMR[minSMRIndex] = 1e9
    return bitsEachLineInBand
#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    ## Testing BitAllocUniform function ##
    print "Start testing BitAllocUniform funciton..."
    print sol.BitAllocUniform(1000,5,5,np.array([3,4,5,6,1]))
    print BitAllocUniform(1000,5,5,np.array([3,4,5,6,1]))
    ## Testing BitAllocConstSNR function ##
    print "Start testing BitAllocConstSNR funciton..."
    blockN = 1024
    fs = 48000.0
    maxMantBits = 5
    n = np.arange(0,blockN)
    testSignal = 0.6*np.cos(2.0*np.pi*420.0*n/fs) + 0.11*np.cos(2.0*np.pi*530.0*n/fs) + 0.10*np.cos(2.0*np.pi*640.0*n/fs) + 0.08*np.cos(2.0*np.pi*840.0*n/fs) + 0.05*np.cos(2.0*np.pi*4200.0*n/fs)+ 0.03*np.cos(2.0*np.pi*8400.0*n/fs)
    scale = 4
    nLines = AssignMDCTLinesFromFreqLimits(blockN/2, fs)
    sfBands = ScaleFactorBands(nLines)
    bitBudget = int(128.0/(fs/1000.0) * blockN/2 - (4+4)*sfBands.nBands - 4)
    MDCTdata = (1<<scale) * MDCT(SineWindow(testSignal),blockN/2,blockN/2)
    SMRs = CalcSMRs(testSignal, MDCTdata, scale, fs, sfBands)
    peakSPLs = np.zeros(sfBands.nBands)

    ## find the max peak SPL in that band ##
    MDCTIntensity = 4.0 * (MDCTdata**2.0)
    MDCT_SPL = SPL(MDCTIntensity)
    for iBand in range(sfBands.nBands):
        maxSPL = MDCT_SPL[sfBands.lowerLine[iBand]]
        for bin in range(sfBands.lowerLine[iBand]+1,sfBands.upperLine[iBand]+1):
            SPL = MDCT_SPL[bin]
            if (SPL > maxSPL):
                maxSPL = SPL
        peakSPLs[iBand] = maxSPL


    print sol.BitAllocConstSNR(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,peakSPLs).astype(int)
    print BitAllocConstSNR(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,peakSPLs)
    ## Testing BitAllocConstMNR function ##
    print "Start testing BitAllocConstMNR function..."
    print sol.BitAllocConstMNR(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMRs).astype(int)
    print BitAllocConstMNR(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMRs)
    ## Testing BitAlloc function ##
    print "Start testing BitAlloc function..."
    print sol.BitAlloc(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMRs)
    print BitAlloc(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMRs)

