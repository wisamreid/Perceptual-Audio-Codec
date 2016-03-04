"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import *  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,LRMS):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    mdctLine=[]

    for iCh in range(codingParams.nChannels):
        rescaleLevel = 1.*(1<<overallScaleFactor[iCh])
        halfN = codingParams.nMDCTLines

        # reconstitute the first halfN MDCT lines of this channel from the stored data
        mdctLine.append(np.zeros(halfN,dtype=np.float64))
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines =codingParams.sfBands.nLines[iBand]
            if bitAlloc[iCh][iBand]:
                mdctLine[iCh][iMant:(iMant+nLines)]=vDequantize(scaleFactor[iCh][iBand], mantissa[iCh][iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iCh][iBand])
            iMant += nLines
        mdctLine[iCh] /= rescaleLevel  # put overall gain back to original level

    # recombine into L and R only
    mdctLineL=mdctLine[0]
    mdctLineR=mdctLine[1]

    for iBand in range(codingParams.sfBands.nBands):
        if LRMS[iBand]:
            lowLine = codingParams.sfBands.lowerLine[iBand]
            highLine = codingParams.sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value

            # Reconstruction, L=M-S and R=M+S
            mdctLineL[lowLine:highLine]=mdctLine[0][lowLine:highLine]-mdctLine[1][lowLine:highLine]
            mdctLineR[lowLine:highLine]=mdctLine[0][lowLine:highLine]+mdctLine[1][lowLine:highLine]

    # IMDCT and window the data for each channel
    dataL = SineWindow( IMDCT(mdctLineL, halfN, halfN) )  # takes in halfN MDCT coeffs
    dataR = SineWindow( IMDCT(mdctLineR, halfN, halfN) )

    data=dataL,dataR

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data

def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    (scaleFactor,bitAlloc,mantissa,overallScaleFactor,LRMS)=EncodeDualChannel(data,codingParams)

    # Huffman here on mantissa

    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,LRMS)

def EncodeDualChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    timeSamples=[]
    mdctTimeSamples=[]
    mdctLines=[]
    maxLine=[]
    overallScale=[]

    for iCh in range(codingParams.nChannels):
        # window data for side chain FFT and also window and compute MDCT
        timeSamples.append(data[iCh])
        mdctTimeSamples.append(SineWindow(data[iCh]))
        mdctLines.append(MDCT(mdctTimeSamples[iCh], halfN, halfN)[:halfN])

        # compute overall scale factor for this block and boost mdctLines using it
        maxLine.append(np.max( np.abs(mdctLines[iCh]) ) )
        overallScale.append(ScaleFactor(maxLine[iCh],nScaleBits) ) #leading zeroes don't depend on nMantBits
        mdctLines[iCh] *= (1<<overallScale[iCh])

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRlr,SMRms,MDCTlr,MDCTms = getStereoMaskThreshold(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands, codingParams)

    bitAlloc=[]
    bitDifference=[]
    scaleFactor=[]
    mantissa=[]

    # perform bit allocation using SMR results
    for iCh in range(codingParams.nChannels):
        ba,bd=BitAlloc(bitBudget, codingParams.extraBits, maxMantBits, sfBands.nBands, sfBands.nLines, SMRlr[iCh])
        bitAlloc.append(ba)
        bitDifference.append(bd)

    for iCh in range(codingParams.nChannels):
        ba,bd=BitAlloc(bitBudget, codingParams.extraBits, maxMantBits, sfBands.nBands, sfBands.nLines, SMRms[iCh])
        bitAlloc.append(ba)
        bitDifference.append(bd)

    LRsum = bitAlloc[0]+bitAlloc[1]
    MSsum = bitAlloc[2]+bitAlloc[3]

    LRMSmdctLines = MDCTlr
    LRMS = np.zeros(sfBands.nBands,dtype=int)

    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1
        if LRsum[iBand] >= MSsum[iBand]:
            LRMSmdctLines[:][lowLine:highLine] = MDCTms[:][lowLine:highLine]
            bitAlloc[0][iBand]=bitAlloc[2][iBand]
            bitAlloc[1][iBand]=bitAlloc[3][iBand]
            LRMS[iBand]=1

    print LRMS

    for iCh in range(codingParams.nChannels):
        # codingParams.extraBits+=bitDifference

        # given the bit allocations, quantize the mdct lines in each band
        scaleFactor.append(np.empty(sfBands.nBands,dtype=np.int32))
        nMant=halfN
        for iBand in range(sfBands.nBands):
            if not bitAlloc[iCh][iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
        mantissa.append(np.empty(nMant,dtype=np.int32))
        iMant=0
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
            nLines= sfBands.nLines[iBand]
            scaleLine = np.max(np.abs( LRMSmdctLines[iCh][lowLine:highLine] ) )
            scaleFactor[iCh][iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iCh][iBand])
            if bitAlloc[iCh][iBand]:
                mantissa[iCh][iMant:iMant+nLines] = vMantissa(LRMSmdctLines[iCh][lowLine:highLine],scaleFactor[iCh][iBand], nScaleBits, bitAlloc[iCh][iBand])
                iMant += nLines
        # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale, LRMS)

if __name__=="__main__":

    execfile('pacfile.py')
