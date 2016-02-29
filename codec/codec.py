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
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs
# from bitalloc import BitAllocUniform
# from bitalloc import BitAllocConstSNR
# from bitalloc import BitAllocConstMNR

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,LRMS):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    mdctLine=[]

    for iCh in range(codingParams.nChannels):
        rescaleLevel = 1.*(1<<overallScaleFactor[iCh])
        halfN = codingParams.nMDCTLines
        # N = 2*halfN
        # vectorizing the Dequantize function call
        # vDequantize = np.vectorize(Dequantize)

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

    # decide LR or MS
    sfBands=codingParams.sfBands
    LRMS=np.zeros(sfBands.nBands,dtype='int')
    L=np.fft.fft(data[0])
    R=np.fft.fft(data[1])
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        LRMS[iBand] = sum(np.power(L[lowLine:highLine],2)-np.power(R[lowLine:highLine],2))<0.8*sum(np.power(L[lowLine:highLine],2)+np.power(R[lowLine:highLine],2))

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)

    # Form L and R mdct
    halfN = codingParams.nMDCTLines
    mdctTimeSamplesL = SineWindow(data[0])
    mdctLinesL = MDCT(mdctTimeSamplesL, halfN, halfN)[:halfN]
    mdctTimeSamplesR = SineWindow(data[1])
    mdctLinesR = MDCT(mdctTimeSamplesR, halfN, halfN)[:halfN]

    # Form M and S
    mdctLinesM=(mdctLinesL+mdctLinesR)/2.
    mdctLinesS=(mdctLinesL-mdctLinesR)/2.
    timeSamplesM=(data[0]+data[1])/2.
    timeSamplesS=(data[0]-data[1])/2.

    # encode M
    (s,b,m,o) = EncodeDualChannel(mdctLinesM,timeSamplesM,codingParams)
    scaleFactor.append(s)
    bitAlloc.append(b)
    mantissa.append(m)
    overallScaleFactor.append(o)

    # encode S
    (s,b,m,o) = EncodeDualChannel(mdctLinesS,timeSamplesS,codingParams)
    scaleFactor.append(s)
    bitAlloc.append(b)
    mantissa.append(m)
    overallScaleFactor.append(o)

    # initialize return variables with L/R as default
    finalScaleFactor=np.array(scaleFactor[0:2])
    finalBitAlloc=np.array(bitAlloc[0:2])
    finalMantissa=np.array(mantissa[0:2])
    finalOverallScaleFactor=np.array(overallScaleFactor[0:2])

    # set MS data per band
    for iBand in range(sfBands.nBands):
        if LRMS[iBand]==True:
            finalScaleFactor=np.array(scaleFactor[2:4])
            finalBitAlloc=np.array(bitAlloc[2:4])
            finalMantissa=np.array(mantissa[2:4])
            finalOverallScaleFactor=np.array(overallScaleFactor[2:4])

    # Huffman here on finalMantissa

    # return results bundled over channels
    return (finalScaleFactor,finalBitAlloc,finalMantissa,finalOverallScaleFactor,LRMS)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    # print halfN
    # N = 2*halfN
    # print N
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
    # vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    mdctTimeSamples = SineWindow(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # # compute SPLs
    # bandPeaks=np.zeros(25)
    # for i in np.arange(sfBands.nBands):
    #     bandPeaks[i]=np.amax(mdctLines[sfBands.lowerLine[i]:sfBands.upperLine[i]])
    # from psychoac import SPL
    # bandPeaks=SPL(bandPeaks)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
    # bitAlloc = BitAllocUniform(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines)
    # bitAlloc = BitAllocConstSNR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, bandPeaks)
    # bitAlloc = BitAllocConstMNR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    # print nMant
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    # print nMant
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            # print len(mdctLines[lowLine:highLine])
            # print scaleFactor[iBand]
            # print nScaleBits
            # print bitAlloc[iBand]
            # print len(mantissa[iMant:iMant+nLines])
            # output = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            # print len(output)
            # print iMant
            # print nMant
            # print nLines
            # print '\n'
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)


def EncodeDualChannel(mdctLines,timeSamples,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)


if __name__=="__main__":

    execfile('pacfile.py')