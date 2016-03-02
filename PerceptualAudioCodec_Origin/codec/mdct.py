"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###

    N=a+b
    n0=(b+1)/2.
    n=np.arange(0,N)
    k=np.arange(0,N/2)
    sig=data

    if not(isInverse):
        
        X=np.zeros(N/2)

        for i in k:
            X[i]=(2./N)*np.sum(sig*np.cos((2*np.pi/N)*(n+n0)*(k[i]+.5)))

        return X
        
    else:

        x=np.zeros(N)

        for i in n:
            x[i]=2.*np.sum(sig*np.cos((2*np.pi/N)*(n[i]+n0)*(k+.5)))

        return x


    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N=a+b
    n0=(b+1)/2.
    n=np.arange(0,N)

    if not(isInverse):
        k=np.arange(0,N/2)
        # Pre-twiddle
        pre=data*np.exp(1j*-2.*np.pi*n/(2.*N))
        # FFT
        fft=np.fft.fft(pre)
        # Post-twiddle
        post=(2./N)*np.real(fft[0:N/2]*np.exp(1j*(-2.*np.pi/N)*n0*(k+(1/2.))))

    else:
        k=np.arange(0,N)
        # Pre-twiddle
        pre=np.hstack((data,-data[::-1]))*np.exp(1j*2.*np.pi*k*n0/N)
        # IFFT
        ifft=np.fft.ifft(pre)
        # Post-twiddle
        post=N*np.real(ifft*np.exp(1j*2*np.pi/(2.*N)*(n+n0)))

    return post
    ### YOUR CODE ENDS HERE ###

def IMDCT(data,a,b):

    ### YOUR CODE STARTS HERE ###
    return MDCT(data,a,b,True)
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    pass # THIS DOES NOTHING

    ### YOUR TESTING CODE ENDS HERE ###

