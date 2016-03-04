"""
-----------------------------------------------------------------------
psychoac.py

Author: Wisam Reid
-----------------------------------------------------------------------
"""

from numpy import *
from window import *
from mdct import *

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """

    minval=Intensity(-30)
    if hasattr(intensity, "__len__"):
        intensity[intensity<minval]=minval
    else:
        if intensity<minval:
            intensity=minval

    spl=96+10*np.log10(intensity)

    if hasattr(spl, "__len__"):
        spl[spl<-30.]=-30.
    else:
        if spl<-30.:
            spl=-30.

    return spl

def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    intensity=10**((spl-96)/10)
    return intensity

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""

    f = clip(f, 10, inf)
    khz = divide(f, 1000.0) # Units

    term1 = 3.64 * (khz ** -0.8)
    term2 = -6.5 * exp(-0.6 * ((khz - 3.3) ** 2))
    term3 = 0.001 * (khz ** 4)

    return term1 + term2 + term3

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """

    khz = divide(f, 1000.0) # Units

    term1 = 13.0 * arctan(khz * 0.76)
    term2 = 3.5 * arctan((khz / 7.5) ** 2)

    return term1 + term2

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self, f, SPL, isTonal = True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.z = Bark(f)
        self.SPL = SPL
        self.isTonal = isTonal

        if isTonal:
            self.drop = 15.0
        else:
            self.drop = 5.5

    def IntensityAtFreq(self, freq):
        """The intensity of this masker at frequency freq"""

        return self.IntensityAtBark(Bark(freq))

    def vIntensityAtFreq(self,fVec):
        """Vectorized intensity of this masker at frequencies in fVec"""

        return self.vIntensityAtBark(Bark(fVec))

    def IntensityAtBark(self, z):
        """The intensity of this masker at Bark location z"""

        dz = z - self.z

        # from hand out
        spreadFunc = ((0.367 * max(self.SPL - 40.0, 0)) * float(dz >= 0) - 27.0) * ((abs(dz)-0.5) * float(abs(dz) > 0.5))
        spl = self.SPL + spreadFunc - self.drop

        return Intensity(spl)

    def vIntensityAtBark(self, zVec):
        """The intensity of this masker at vector of Bark locations zVec"""

        dzVec = subtract(zVec, self.z)

        slope = greater_equal(dzVec, 0)
        leveling = (0.367 * max(self.SPL - 40.0, 0))

        spreadFunc = ((slope * leveling) - 27.0) * ((absolute(dzVec) - 0.5) * greater(absolute(dzVec), 0.5))

        splVec = self.SPL + spreadFunc - self.drop

        return Intensity(splVec)

cbFreqLimits = (100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 24000.0)

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """

    # shift by 1/2
    mdct_lines = (arange(nMDCTLines) + 0.5) / nMDCTLines * (sampleRate / 2)

    lower = 0
    upper = 0
    assignments = []

    for limit in flimit:

        if limit >= (sampleRate/2.0):

            upper = sampleRate / 2.0

        else:

            upper = limit

        # truncate mdct_lines
        mdct_lines_in_band = mdct_lines[mdct_lines<=upper]
        mdct_lines_in_band = mdct_lines_in_band[mdct_lines_in_band>lower]

        assignments.append(len(mdct_lines_in_band))
        lower = upper

    return assignments

def findpeaks(Xwdb, sampleRate, N):

    peaks = []
    freqs = []

    length = size(Xwdb)

    # find peaks and order from max amplitude to min
    for sample in range(1,length-1):

        if (abs(Xwdb[sample]) > abs(Xwdb[sample-1]) and abs(Xwdb[sample]) > abs(Xwdb[sample+1]) and 10.0* log10(abs(Xwdb[sample]))>-30.0):

            peaks = np.append(peaks,Xwdb[sample])
            freqs = np.append(freqs,sample)

    # peaks = peaks.astype(int)
    peaks = absolute(peaks).astype(int)
    freqsIndex = asarray(freqs).astype(int)

    # parabolic interpolation
    estimateFreqs = []
    estimateAmp = []

    for idx in range(0,len(freqs)):

        a = abs(Xwdb[freqs[idx]-1])
        b = abs(Xwdb[freqs[idx]])
        r = abs(Xwdb[freqs[idx]+1])
        p = (1/2)*(a-r)/(a+r-2*b)
        A = b-(a-r)*(p/4)
        estimateFreqs = append(estimateFreqs,(freqs[idx]+p)*(sampleRate/N))
        estimateAmp = append(estimateAmp,A)

    return estimateAmp, estimateFreqs, freqsIndex

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self, nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """

        self.nBands = len(nLines)
        self.nLines = array(nLines)
        self.lowerLine = append(0, cumsum(nLines)[:-1])
        self.upperLine = self.lowerLine + nLines
        self.upperLine = add(self.nLines, subtract(self.lowerLine, 1))

###########################################################################
############################## Stereo Coding ##############################
###########################################################################

### Calculate Masking level difference ###
def MLD(z):
    """
    Calculate the masking level difference factors for z

    Arguments:

            z: Bark frequencies

    Returns:

            An array of MLD factors
    """
    a = 1.25
    offset = 2.5

    MLD = np.zeros_like(z)

    MLD = np.power(10.0, a * (1 - np.cos(np.pi * (np.minimum(z, 15.5)/15.5)) - offset))

    # normalize
    MLD = MLD/np.amax(MLD)

    return MLD

def MLD_F(f):
    """
    Calculate the masking level difference factors for f

    Arguments:

            f: Array of linear frequencies

    Returns:

            MLD: Array of MLD factors
    """
    a = 1.25
    offset = 2.5
    cutoff = 3000.

    MLD = np.zeros_like(f)

    MLD = np.power(10.0, a * (1 - np.cos(np.pi * (np.minimum(f, cutoff)/cutoff) ) - offset))

    # normalize
    MLD = MLD/np.amax(MLD)

    return MLD

def calcBTHR(data, MDCTdata, MDCTscale, sampleRate, sfBands, noDrop):
    """
    ####### Helper function:  getStereoMaskThreshold #######

    data:

            data: Audio (per channel) [L, R]
            MDCTdata: MDCT lines (per channel)
            MDCTscale: scale factor (per channel)
            sampleRate: FS
            sfBands: Scale factor bands (shared)

    Return:

            Masked Threshold evaluated at MDCT lines.
    """

    N = len(data)
    nMDCTLines = len(MDCTdata)
    X_fft = fft.fft(HanningWindow(data))[0:N/2]
    alpha = 1.0

    masked_intensity = zeros_like(MDCTdata)

    # shift by 0.5 samples
    MDCTFreqs = sampleRate / 2.0 / nMDCTLines * (arange(0, nMDCTLines) + 0.5)

    # threshold in quiet
    threshold_in_quiet = Intensity(Thresh(MDCTFreqs))**alpha

    # Using JOS code written for 320A
    estimated_peak_amplitudes, estimated_peak_frequencies, index = findpeaks(X_fft,sampleRate,N)

    BW = 3
    masker_spl = zeros(len(index), dtype=float64)
    num_peaks = len(index)

    # aggregate intensity across the peaks, create maskers, sum maskers
    for i in range(num_peaks):
        masker_spl[i] = SPL((8.0 / 3.0 * 4.0 / (N ** 2.0)) * sum(abs(X_fft[index[i] - BW:index[i] + BW])**2.0))
        maskers = Masker(estimated_peak_frequencies[i],masker_spl[i],True)
        if noDrop:
            maskers.drop=0
        masked_intensity += (maskers).vIntensityAtFreq(MDCTFreqs)**alpha

    masked_intensity = (masked_intensity + threshold_in_quiet)**(1.0/alpha)

    return  SPL(masked_intensity)

def calcStereoSMR(stereoThreshold, mdctSPL, sfBands):
    """
    ####### Helper function:  getStereoMaskThreshold #######

    Calculates the max SMR (per band) for M/S or L/R masking curves

    Arguments:

            stereoThreshold: stereo masking curve (per channel: M/S or L/R)
            mdctSPL: MDCT data masking curve in SPL (per channel)
            sfBands: Scale factor bands (shared)

    Returns:

            Array of Max SMRs: SMRs[channel][band]

    """

    # two channels
    numChannels = 2
    SMRs = []

    for channel in range(numChannels):

        SMRs.append([])
        # for each band calculate max SMR
        for band in range(sfBands.nBands):

            lower = sfBands.lowerLine[band]
            upper = sfBands.upperLine[band]+1

            # get mask for this band
            mask = stereoThreshold[channel][lower:upper]
            x = mdctSPL[channel][lower:upper]
            # SMR for whole band
            bandSMR = x - mask

            # max
            if len(bandSMR) == 0:

                SMRs[channel].append(-96.0)

            else:

                SMRs[channel].append(np.amax(bandSMR))

    return SMRs

def getStereoMaskThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands, codingParams):
    """
    Calculates the stereo masking theshold for M/S or L/R

    Arguments:

            data: Audio data (per channel) [L, R]
            MDCTdata: MDCT lines (per channel)
            MDCTscale: scale factor (per channel)
            sampleRate: FS
            sfBands: Scale factor bands (shared)
            LRMS: 0 --> transmitting L/R 1, --> transmitting M/S

    Returns:

            SMR[channel][nBands] Masked Threshold evaluated at MDCT lines.
    """

    # Comment and uncomment for block by block Threshold printing (Every tenth block)
    # codingParams.curBlock = (codingParams.curBlock+1)%10
    if codingParams.curBlock==1:
        print_thresh = True
    else:
        print_thresh = False

    # Comment and uncomment Threshold printing (Every block)
    # print_thresh = True
    # print_thresh = False

    ################ L/R SMR calculation ################

    # calculate MDCT SPL for L/R
    MDCT_Spl_L = SPL(4.*MDCTdata[0]**2)-(6.02*MDCTscale[0])
    MDCT_Spl_R = SPL(4.*MDCTdata[1]**2)-(6.02*MDCTscale[1])
    MDCT_Spl_LR = [MDCT_Spl_L, MDCT_Spl_R]

    # calculate basic thresholds for LR
    BTHR_L = calcBTHR(data[0], MDCTdata[0], MDCTscale[0], sampleRate, sfBands, False) # Has drop
    BTHR_R = calcBTHR(data[1], MDCTdata[1], MDCTscale[1], sampleRate, sfBands, False)
    THR_LR = [BTHR_L, BTHR_R]

    ################ M/S SMR calculation ################

    # transform time domain L/R data into M/S
    data_MS = [(data[0] + data[1]) / 2.0, (data[0] - data[1]) / 2.0]
    # transform MDCT L/R data into M/S
    MDCT_data_MS = [(MDCTdata[0] + MDCTdata[1]) / 2.0, (MDCTdata[0] - MDCTdata[1]) / 2.0]

    # calculate MDCT SPL for M/S
    MDCT_Spl_M = SPL(4.*MDCT_data_MS[0]**2)-(6.02*MDCTscale[0])
    MDCT_Spl_S = SPL(4.*MDCT_data_MS[1]**2)-(6.02*MDCTscale[1])
    MDCT_Spl_MS = [MDCT_Spl_M, MDCT_Spl_S]

    # calculate basic thresholds for MS
    BTHR_M = calcBTHR(data_MS[0], MDCT_data_MS[0], MDCTscale[0], sampleRate, sfBands, False) # Has drop
    BTHR_S = calcBTHR(data_MS[1], MDCT_data_MS[0], MDCTscale[1], sampleRate, sfBands, False)
    BTHR_M_MLD = calcBTHR(data_MS[0], MDCT_data_MS[0], MDCTscale[0], sampleRate, sfBands, True) # No drop
    BTHR_S_MLD = calcBTHR(data_MS[1], MDCT_data_MS[0], MDCTscale[1], sampleRate, sfBands, True)
    BTHR_MS = [BTHR_M, BTHR_S]

    ################ calculate MLD ################

    # MDCT freqs [Hz]
    MDCT_freqs = (((np.arange(len(MDCTdata[0])) + 0.5) / len(MDCTdata[0])) * (sampleRate / 2.0))

    # get MLDs
    mld = MLD_F(MDCT_freqs)

    # drop it low shorty
    MLD_M = BTHR_M_MLD * mld
    MLD_S = BTHR_S_MLD * mld

    # calculate actual threshold for MS
    THR_MS = [np.maximum(BTHR_M, np.minimum(BTHR_S, MLD_S)), np.maximum(BTHR_S, np.minimum(BTHR_M, MLD_M))]

    # get max SMRs for L/R
    SMR_LR = calcStereoSMR(THR_LR, MDCT_Spl_LR, sfBands)

    # get max SMRs for M/S
    SMR_MS = calcStereoSMR(THR_MS, MDCT_Spl_MS, sfBands)

    ################ Begin Threshold Plots ################
    if print_thresh:

        plt.figure(1)
        # plot L/R SPL, curve, and SMR

        plt.subplot(211)
        plt.title('SPL of MDCT, LR masking curve and SMRs')
        pltMDCT, = plt.semilogx( MDCT_Spl_LR[0], 'k')
        pltThresh, = plt.semilogx(THR_LR[0], 'c--', linewidth=3.0)
        pltSMRL = plt.bar(sfBands.lowerLine, SMR_LR[0], sfBands.nLines, alpha=0.5, color="green")
        plt.legend([pltMDCT, pltThresh], ["signal MDCT SPL", "Actual threshold"])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('L channel: SPL [dB]')
        plt.xlim((0,1024))

        plt.subplot(212)
        plt.semilogx(MDCT_Spl_LR[1], 'k')
        plt.semilogx(THR_LR[1], 'c--', linewidth=3.0)
        pltSMRR = plt.bar(sfBands.lowerLine, SMR_LR[1], sfBands.nLines, alpha=0.5, color="green")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('R channel: SPL [dB]')
        plt.xlim((0,1024))

        plt.figure(2)
        # plot M/S SPL, curve, and SMR

        plt.subplot(211)
        plt.title('SPL of MDCT, MS masking curve and SMRs')
        pltMDCT, = plt.semilogx(MDCT_Spl_MS[0], 'k')
        pltBthresh, = plt.semilogx( BTHR_MS[0], 'r' , linewidth=3.0)
        pltThresh, = plt.semilogx( THR_MS[0], 'c--', linewidth=3.0)
        pltSMRM = plt.bar(sfBands.lowerLine, SMR_MS[0], sfBands.nLines, alpha=0.5, color="green")
        plt.legend([pltMDCT, pltBthresh, pltThresh], ["signal MDCT SPL", "Basic threshold", "Actual Threshold"])
        plt.xlabel('Freq (Hz)')
        plt.ylabel('M channel: SPL (dB)')
        plt.xlim((0,1024))

        plt.subplot(212)
        plt.semilogx( MDCT_Spl_MS[1], 'k' )
        plt.semilogx(BTHR_MS[1], 'r' , linewidth=3.0)
        plt.semilogx(THR_MS[1], 'c--', linewidth=3.0 )
        pltSMRS = plt.bar(sfBands.lowerLine, SMR_MS[1], sfBands.nLines, alpha=0.5, color="green")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('S channel: SPL (dB)')
        plt.xlim((0,1024))

        plotBars = [pltSMRL,pltSMRR,pltSMRM,pltSMRS, print_thresh]

    else:

        plotBars = [False,False,False,False,False]

    ################ End Threshold Plots ################


    return SMR_LR,SMR_MS,MDCTdata,MDCT_data_MS, plotBars

if __name__ == '__main__':

    test_problem1a = False
    test_problem1b = False
    test_problem1c = False
    test_problem1d = False
    test_problem1e = False
    test_problem1f = False
    test_problem1g = False
    test_mld = False
    test_stereo_masks = False

    #### construct input signal ####
    FS = 48000.0
    N = 1024
    n = arange(N, dtype=float)
    x_L = zeros_like(n)
    x_R = zeros_like(n)

    freqs_L = (420, 530, 640, 840, 4200, 8400)
    amps_L = (0.60, 0.11, 0.10, 0.08, 0.05, 0.03)

    freqs_R = (420, 530, 640)
    amps_R = (0.60, 0.11, 0.10)

    for i in range(len(freqs_L)):
        x_L += amps_L[i] * cos(2 * pi * freqs_L[i] * n / FS)

    for i in range(len(freqs_R)):
        x_R += amps_R[i] * cos(2 * pi * freqs_R[i] * n / FS)

    sfBands = ScaleFactorBands( AssignMDCTLinesFromFreqLimits(N, FS) )

    if test_stereo_masks:
        # MDCT Left
        MDCT_data_Sine_L = MDCT(SineWindow(x_L), N / 2, N / 2)

        # MDCT Right
        MDCT_data_Sine_R = MDCT(SineWindow(x_R), N / 2, N / 2)

        mdctLines = [MDCT_data_Sine_L,MDCT_data_Sine_R]
        x=x_L,x_R

        maxLine = []
        overallScale = []
        from quantize import ScaleFactor
        from numpy import fft
        for iCh in range(2):
            # compute overall scale factor for this block and boost mdctLines using it
            maxLine.append(np.max( np.abs(mdctLines[iCh]) ) )
            overallScale.append(ScaleFactor(maxLine[iCh],4) ) #leading zeroes don't depend on nMantBits
            mdctLines[iCh] *= (1<<overallScale[iCh])

        # calc LRMS
        LRMS=np.zeros(sfBands.nBands,dtype='int')
        L=np.fft.fft(x[0])
        R=np.fft.fft(x[1])
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
            LRMS[iBand] = sum(np.power(L[lowLine:highLine],2)-np.power(R[lowLine:highLine],2))<0.8*sum(np.power(L[lowLine:highLine],2)+np.power(R[lowLine:highLine],2))

        # create masks
        getStereoMaskThreshold(x, mdctLines, overallScale, FS, sfBands, LRMS, [])

    if test_problem1b:

        X = abs(fft(HanningWindow(x)))[0:N / 2]
        f = fftfreq(N, 1 / FS)[0:N / 2]
        X_spl = SPL(8.0 / 3.0 * 4.0 / N ** 2 * abs(X) ** 2)

        n = arange(0,N)

        f = (FS/2.0) * arange(0.0,N/2) / (N/2)
        f[0] = 20.0

        estimate_amp, estimate_freqs, freqs_index = findpeaks(X,FS,N)

        print "Peaks: \n"
        print  freqs
        print "Estimates: \n"
        print  estimate_freqs

        signal_SPL = SPL( 4.0/(N**2.0 * 3.0 / 8.0) * abs(X)**2.0)

        figure(figsize=(14, 6))
        semilogx(f, signal_SPL, 'b')
        semilogx(f[freqs_index], signal_SPL[freqs_index], 'ro')
        xlim(50, FS / 2)
        ylim(-50, 100)
        ylabel("SPL (dB)")
        xlabel("Frequency (Hz)")
        title("Signal SPL and peaks N = %i" % N) # Dynamic Title
        grid()
        show()

    if test_problem1c:

        X = abs(fft(HanningWindow(x)))[0:N / 2]
        f = fftfreq(N, 1 / FS)[0:N / 2]
        X_spl = SPL(8.0 / 3.0 * 4.0 / N ** 2 * abs(X) ** 2)


        figure(figsize=(14, 6))
        semilogx(f, Thresh(f), 'g', label='Threshold in quiet')
        semilogx(f, X_spl, 'm', label='FFT SPL')
        xlabel('Frequency (Hz)')
        ylabel('SPL (dB)')
        xlim(50, FS / 2)
        ylim(-50, 250)
        grid()
        legend()
        title('Input SPL and Threshold in Quiet')
        show()

    if test_problem1d:

        lower_freqs = [0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0]
        print "Freqs to Barks:\n"
        print Bark(lower_freqs)

    if test_problem1e:

        # FFT
        X = abs(fft(HanningWindow(x)))[0:N / 2]
        f = fftfreq(N, 1 / FS)[0:N / 2]
        X_spl = SPL(8.0 / 3.0 * 4.0 / N ** 2 * abs(X) ** 2)

        # MDCT
        pow_KBD = 1.0 / N * sum(SineWindow(ones(N)))
        MDCT_data_KBD = MDCT(SineWindow(x), N / 2, N / 2)
        MDCT_scale = zeros_like(MDCT_data_KBD)
        mdct_PSD_KBD = (abs((2.0 ** MDCT_scale) * MDCT_data_KBD * N / 2) ** 2)
        mdct_SPL_KBD = SPL(8.0 * mdct_PSD_KBD / ((N ** 2) * pow_KBD))
        MDCT_freqs = (arange(N / 2) + 0.5) * FS / float(N)

        # plot threshold in quiet, MDCT-KBD and FFT-Hann spectrum
        figure(figsize=(14, 6))
        semilogx(f, Thresh(f), 'g', label='Threshold in quiet')
        semilogx(f, X_spl, 'm', label='FFT - Hanning Window')
        semilogx(MDCT_freqs, mdct_SPL_KBD, 'r', label='MDCT - Sine Window')
        xlabel('Frequency (Hz)')
        ylabel('SPL (dB)')
        xlim(50, FS / 2)
        ylim(-50, 150) # change for resolution

        # create masks
        total_intensity = zeros_like(f)

        for i in range(len(freqs)): # create a masker at every peak
            masker = Masker(freqs[i], SPL(amps[i] ** 2), True)
            plot(f + 0.5, SPL(masker.vIntensityAtBark(Bark(f))), 'b--', linewidth=1.0)
            total_intensity += masker.vIntensityAtBark(Bark(f))

        total_intensity += Intensity(Thresh(f)) # add threshold in quiet
        mask_threshold = SPL(total_intensity)

        plot(f + 0.5, SPL(masker.vIntensityAtBark(Bark(f))), 'b--', linewidth=1.0, label='Masker(s)')# plot again for legend
        plot(f + 0.5, mask_threshold, 'c', linewidth=3.0, label='Total Masked Threshold')
        ylabel('SPL (dB)')
        grid()
        title('Masking Curve: N = 1024')
        legend(loc=2)
        show()

    if test_problem1f:


        # FFT
        X = abs(fft(HanningWindow(x)))[0:N / 2]
        f = fftfreq(N, 1 / FS)[0:N / 2]
        X_spl = SPL(8.0 / 3.0 * 4.0 / N ** 2 * abs(X) ** 2)

        # MDCT
        pow_KBD = 1.0 / N * sum(KBDWindow(ones(N)))
        MDCT_data_KBD = MDCT(KBDWindow(x), N / 2, N / 2)
        MDCT_scale = zeros_like(MDCT_data_KBD)
        mdct_PSD_KBD = (abs((2.0 ** MDCT_scale) * MDCT_data_KBD * N / 2) ** 2)
        mdct_SPL_KBD = SPL(8.0 * mdct_PSD_KBD / ((N ** 2) * pow_KBD))
        MDCT_freqs = (arange(N / 2) + 0.5) * FS / float(N)

        # plot threshold in quiet, MDCT-KBD and FFT-Hann spectrum
        figure(figsize=(14, 6))
        semilogx(f, Thresh(f), 'g', label='Threshold in quiet')
        semilogx(f, X_spl, 'm', label='FFT - Hanning Window')
        semilogx(MDCT_freqs, mdct_SPL_KBD, 'r', label='MDCT - KBD Window')
        xlabel('Frequency (Hz)')
        ylabel('SPL (dB)')
        xlim(50, FS / 2)
        ylim(-50, 150) # change for resolution

        # create masks
        total_intensity = zeros_like(f)

        for i in range(len(freqs)): # sum maskers at every peak
            masker = Masker(freqs[i], SPL(amps[i] ** 2), True)
            total_intensity += masker.vIntensityAtBark(Bark(f))

        total_intensity += Intensity(Thresh(f)) # add threshold in quiet
        mask_threshold = SPL(total_intensity)

        scaleplt = vlines(cbFreqLimits, -50, 350, colors='y',linewidth=2.0, alpha=0.75)
        CBs = array([50] + [ l for l in cbFreqLimits ])
        CBs = sqrt(CBs[1:] * CBs[:-1])

        for i, val in enumerate(CBs, start=1):
            scaletextplt = text(val, -40, str(i), horizontalalignment='center')

        plot(f + 0.5, mask_threshold, 'c', linewidth=3.0, label='Total Masked Threshold')
        ylabel('SPL (dB)')
        grid()
        title('Band Boundaries')
        legend(loc=2)
        show()

    if test_problem1g:

        nMDCTLines = N / 2
        scaleFactor = 6
        nLines = AssignMDCTLinesFromFreqLimits(nMDCTLines, FS)

        sfb = ScaleFactorBands(nLines)

        MDCT_data_KBD = MDCT(KBDWindow(x), N / 2, N / 2) * (2 ** scaleFactor)

        SMR_mdct_KBD = CalcSMRs(x, MDCT_data_KBD, scaleFactor, FS, sfb)
        SMR_mdct_KBD = array(SMR_mdct_KBD)


        print "SMRs For Table"
        print SMR_mdct_KBD

    if test_mld:

        print "---- Testing MLD function ----"
        plt.figure(1)
        mld = MLD(Bark(np.linspace(0, 20000, 100)))
        plt.plot(mld)
        plt.yscale('linear')
        plt.xlabel('Bark Frequency [z]')
        plt.ylabel('MLD(z)')
        plt.xlim(0,16.0)
        plt.title('MLD Factor in z')
        plt.show()

        plt.figure(2)
        mld = MLD_F(np.linspace(0, 20000, 200))
        plt.semilogx(np.linspace(0, 20000, 200),mld)
        plt.yscale('linear')
        plt.xlabel('Linear Frequency [f]')
        plt.ylabel('MLD(f)')
        plt.xlim(0,5000)
        plt.title('MLD Factor in f')
        plt.show()

        print mld[0:40]
        # print shape(mld)


    # execfile('pacfile.py')
