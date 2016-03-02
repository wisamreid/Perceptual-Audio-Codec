import numpy as np
from window import *

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    minval=Intensity(-30)
    if hasattr(intensity, "__len__"):
        intensity[intensity<minval]=minval
    
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
    fkHz=np.divide(f,1000.)

    if hasattr(fkHz, "__len__"):
        fkHz[fkHz==0]=.001

    threshspl=3.64*(fkHz)**-.8 - \
              6.5*np.exp(-.6*((fkHz-3.3)**2)) + \
              (10**-3)*(fkHz)**4

    return threshspl

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    fkHz=np.divide(f,1000.)

    z = 13.*np.arctan(.76*fkHz) + 3.5*np.arctan((fkHz/7.5)**2)

    return z

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.z=Bark(f)

        self.f=f

        self.SPL=SPL

        if isTonal:
            self.drop=15.
        else:
            self.drop=5.5

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        intensity = self.IntensityAtBark(Bark(freq))
        return intensity

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        dz = z - self.z

        if abs(dz)<.5:
            curve=self.SPL
        elif dz<-.5:
            curve=-27.*(abs(dz)-.5)+self.SPL
        elif dz>.5:
            curve=(-27+.37*max(self.SPL-40,0))*(dz-.5)+self.SPL

        intensity=Intensity(curve-self.drop)

        return intensity

    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        dz = zVec - self.z
        curve=np.zeros_like(zVec)

        curve[abs(dz)<.5]=self.SPL
        curve[dz<-.5]=-27.*(abs(dz[dz<-.5])-.5)+self.SPL
        curve[dz>.5]=(-27+.37*max(self.SPL-40,0))*(dz[dz>.5]-.5)+self.SPL

        intensity=Intensity(curve-self.drop)

        return intensity

class FindPeaksPara: # IMPLEMENT CRITICAL BANDS

    def __init__(self,Xwdb,maxPeaks,fs,win,N):

        allPeaks = []
        for i in np.arange(1,len(Xwdb)-1):
            if Xwdb[i-1]<Xwdb[i] and Xwdb[i]>Xwdb[i+1]:
                idxpeak=np.array([i,Xwdb[i]])
                allPeaks.append(idxpeak)

        if allPeaks == []:
            height=np.array([0])
            freqs=np.array([0])
        else:
            allPeaks = np.array(allPeaks)
            posPeaks = allPeaks[0:len(allPeaks)/2,]
            peaks=posPeaks[posPeaks[:,1].argsort()[::-1]]

            idx=0
            a=0
            b=0
            c=0
            p=0
            location=np.zeros(len(peaks))
            height=np.zeros(len(peaks))

            for i in np.arange(len(peaks)):
                idx=peaks[i,0]

                #parabolic interpolation
                a=Xwdb[idx-1]
                b=Xwdb[idx]
                c=Xwdb[idx+1]
                
                p=1/2.*(a-c)/(a-2.*b+c)

                location[i]=idx+p
                height[i]=b-1/4.*(a-c)*p

            freqs = fs*location/N

        self.height=height
        self.freqs=freqs

class FindPeaks:

    def __init__(self,Xwdb,fs,N):

        allPeaks = []
        for i in np.arange(1,len(Xwdb)-1):
            if Xwdb[i-1]<Xwdb[i] and Xwdb[i]>Xwdb[i+1]:
                allPeaks.append(i)

        if allPeaks == []:
            height = np.array([0])
            freqs = np.array([0])
        else:
            allPeaks = np.array(allPeaks)
            posPeaks = allPeaks[0:len(allPeaks)/2]
            peaks = posPeaks[posPeaks.argsort()[::-1]]

            idx = 0
            a=0
            b=0
            c=0
            location = np.zeros(len(peaks))
            height = np.zeros(len(peaks))

            for i in np.arange(len(peaks)):
                idx = peaks[i]

                a=Intensity(Xwdb[idx-1])
                b=Intensity(Xwdb[idx])
                c=Intensity(Xwdb[idx+1])

                location[i] = (a*(idx-1)+b*idx+c*(idx+1)) / (a+b+c)
                height[i] = SPL(a+b+c)

            freqs = fs*location/N

        self.height=height
        self.freqs=freqs

# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,24000]

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    
    N=nMDCTLines *2
    mdctfreqs=(np.linspace(0,N/2-1,nMDCTLines)+.5)*sampleRate/N

    curband=0
    linesperband=np.zeros(25,dtype='int')

    for i in mdctfreqs:
        if i < flimit[curband]:
            linesperband[curband]+=1
        else:
            linesperband[curband+1]+=1
            curband+=1

    return linesperband

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """

        self.nBands=len(nLines)
        self.lowerLine=np.hstack(([0],np.cumsum(nLines)[:len(nLines)-1])).astype(np.uint16)
        self.upperLine=np.cumsum(nLines).astype(np.uint16)-1
        self.nLines=np.array(nLines).astype('uint16')
        # self.nLines=self.upperLine-self.lowerLine+1

def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N=len(data)
    X=np.fft.fft(HanningWindow(data))

    hann=HanningWindow(np.ones(N))
    w2hann=sum(hann**2)/N

    Xspl=SPL(4.*abs(X)**2/((N**2)*w2hann))
    # Xspl=SPL((32*abs(X)**2)/(3*N**2))

    # IMPLEMENT RECURSIVE MASK CHECKING
    # posfreqs=np.linspace(0,sampleRate/2,N/2)
    # mdctfreqs=posfreqs+((sampleRate/2.)/len(MDCTdata))
    spacing = (sampleRate/2)/(N/2)
    mdctfreqs = spacing*np.linspace(.5,N/2+.5,N/2)

    # IMPLEMENT CRITICAL BANDS HERE
    peakinfo=FindPeaks(Xspl,sampleRate,N)

    maskers=[]
    # curvesint=[]
    totmask=np.zeros(len(mdctfreqs))
    for i in np.arange(len(peakinfo.height)):
        maskers.append(Masker(peakinfo.freqs[i],peakinfo.height[i]))
        # curvesint.append(maskers[i].vIntensityAtBark(Bark(mdctfreqs)))
        # totmask+=curvesint[i]
        totmask+=maskers[i].vIntensityAtBark(Bark(mdctfreqs))

    threshint=Intensity(Thresh(mdctfreqs))
    totmask+=threshint

    maskingcurve=SPL(totmask)

    return maskingcurve

def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency lines for the data
                            in data which have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  is an overall scale factor for the set of MDCT
                            frequency lines
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Sums their masking curves with the hearing threshold at each MDCT
                frequency location to the calculate absolute threshold at those
                points. Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """

    maskingCurve=getMaskedThreshold(data,MDCTdata,MDCTscale,sampleRate,sfBands)
    
    N=len(data)
    # sine=SineWindow(np.ones(N))
    # w2sine=sum(sine**2)/N

    # mdctspl=SPL(2.*abs(MDCTdata)**2/(w2sine))
    mdctspl=SPL(4.*MDCTdata**2)-(6.02*MDCTscale)

    allSMRs=mdctspl-maskingCurve

    # # YO ROSS UNCOMMENT THIS BLOCK TO SEE SOME GOOOOOOOD SHIT ON PROBLEM 1G
    # import matplotlib.pyplot as plt
    # # posfreqs=np.linspace(0,sampleRate/2,N/2)
    # # mdctfreqs=posfreqs+((sampleRate/2.)/len(MDCTdata))
    # spacing = (sampleRate/2)/(N/2)
    # mdctfreqs = spacing*np.linspace(.5,N/2+.5,N/2)
    # plt.figure(99)
    # plt.semilogx(mdctfreqs,mdctspl)
    # plt.semilogx(mdctfreqs,maskingCurve,label='mycurve')
    # # plt.plot(mdctfreqs,maskingCurve,label='mycurve')
    # import psychoac_ as psysol
    # solnmaskCurve=psysol.getMaskedThreshold(data,MDCTdata,MDCTscale,sampleRate,sfBands)
    # plt.semilogx(mdctfreqs,solnmaskCurve,label='solncurve')
    # # plt.plot(mdctfreqs,solnmaskCurve,label='solncurve')
    # for i in cbFreqLimits:
    #     plt.axvline(i,-35,110)
    # plt.ylim((-35,110))
    # plt.xlim((40,20000))
    # plt.legend(loc=3)
    # plt.show(block=False)
    # # END GRAPH BLOCK

    SMRs=np.zeros(sfBands.nBands)

    for i in np.arange(sfBands.nBands): # VECTORIZE THIS BROOOO
        SMRs[i]=np.amax(allSMRs[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])

    # SMRs=np.amax(allSMRs)
    
    return SMRs


#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    execfile("HW4.py")
