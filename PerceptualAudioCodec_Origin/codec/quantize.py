"""
quantize.py -- routines to quantize and dequantize floating point aNumues
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import matplotlib.pyplot as plt
import pylab


# Testing Booleans
test_uniform = False
test_uniform_vectorized = False
test_fp = False
test_bfp = False
test_bfp_vectorized = False


# abbreviations
pi = np.pi
zeros = np.zeros
absolute = np.absolute
signbit = np.signbit
empty = np.empty
uint64 = np.uint64
float64 = np.float64
bool = np.bool
arange = np.arange
sqrt = np.sqrt
sin = np.sin
sum = np.sum
log10 = np.log10

# Inputs for table
chartInputs = np.array([-1.0,-0.98,-0.51,-0.02,0.0,0.05,0.41,0.82,0.95,1.0],float64)

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """

    inputVal = aNum # Copy Value
    signBitMask = 1 << nBits - 1 # 2^(R-1)
    largestVal = (signBitMask << 1) - 1.0 # 2^R - 1 (Largest Value)

    # Edge Case
    if nBits <= 0:
        return 0

    if absolute(inputVal) >= 1:
        #The overload level of the quantizer should be 1.0
        aQuantizedNum = signBitMask - 1 # clip to 2^(R-1)-1
    else:
        # absolute(code) = int((2^R-1)*absolute(number)+1)/2)
        aQuantizedNum = int((largestVal * absolute(inputVal) + 1.0) / 2.0)

    # if negative flip the sign bit
    if inputVal < 0:
        aQuantizedNum += signBitMask

    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    signBitMask = 1 << nBits - 1 # 2^(R-1)
    largestVal = (signBitMask << 1) - 1.0 # 2^R - 1 (Largest Value)

    # Edge Case
    if nBits <= 0:
        return 0

    if aQuantizedNum & signBitMask:
        aQuantizedNum -= signBitMask
        # absolute(number) = 2 * absolute(code) / (2^R-1)
        aNum = 2.0 * absolute(aQuantizedNum) / largestVal
        aNum *= -1.0
    else:
        # absolute(number) = 2 * absolute(code) / (2^R-1)
        aNum = 2.0 * absolute(aQuantizedNum) / largestVal

    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    inputValVec = aNumVec.copy() # Copy Values
    aQuantizedNumVec = empty(len(inputValVec), dtype=uint64) # empty quantized vector
    signBitMask = 1 << nBits - 1 # 2^(R-1)
    largestVal = (signBitMask << 1) - 1.0 # 2^R - 1 (Largest Value)

    # Edge Case
    if nBits <= 0:
        return np.zeros(N, dtype=uint64)

    # vectorized sign bit
    sign = signbit(inputValVec)

    inputValVec = absolute(inputValVec)

    # absolute(code) = int((2^R-1)*absolute(number)+1)/2)
    aQuantizedNumVec[inputValVec < 1] = ((inputValVec[inputValVec < 1] * largestVal + 1.0) / 2.0).astype(uint64)
    # clip to 2^(R-1)-1
    aQuantizedNumVec[inputValVec >= 1] = signBitMask - 1

    # if negative flip the sign bit
    aQuantizedNumVec[sign] += signBitMask

    return aQuantizedNumVec

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """

    # Edge Case
    if nBits <= 0:
        return zeros(N, dtype=float64)

    inputValVec = aQuantizedNumVec.copy() # Copy Values
    signBitMask = 1 << nBits - 1 # 2^(R-1)
    largestVal = (signBitMask << 1) - 1.0 # 2^R - 1 (Largest Value)

    negativeVals = aQuantizedNumVec & signBitMask == signBitMask

    sign = zeros(len(aQuantizedNumVec), dtype=bool)

    sign[negativeVals] = True
    inputValVec[negativeVals] -= signBitMask

    # absolute(number) = 2 * absolute(code) / (2^R-1)
    aNumVec = 2.0 * inputValVec / largestVal

    aNumVec[sign] = -aNumVec[sign]

    return aNumVec

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction
    aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros

    # Edge Case
    if nScaleBits < 0:
        nScaleBits = 0
    if nMantBits <= 0:
        return 0

    scale = 0

    largestScale = (1 << nScaleBits) - 1 # 2^Rs - 1
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    # bit mask to count zeros
    zeroBitMask = 1 << R - 1

    # chop off sign bit after quantization
    aQuantizedNum = (QuantizeUniform(absolute(aNum), R)) << 1
    # count the number of leading zeros in the uniformly quantized code
    while scale < largestScale and zeroBitMask & aQuantizedNum == 0:
        aQuantizedNum <<= 1
        scale += 1

    return scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    signBitMask = 1 << nMantBits - 1

    # Edge Cases
    if nMantBits <= 0:
        return 0

    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    mantissa = QuantizeUniform(absolute(aNum), R) << scale + 1

    if scale < largestScale:
        mantissa -= 1 << R - 1
        mantissa <<= 1

    mantissa >>= R - nMantBits + 1

    # if negative flip the sign bit
    if aNum < 0:
        mantissa += signBitMask

    return mantissa

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    signBitMask = 1 << nMantBits - 1

    # Edge Case
    if nMantBits <= 0:
        return 0.0
    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    if mantissa & signBitMask:
        sign = 1
        mantissa -= signBitMask
    else:
        sign = 0

    if scale < largestScale:
        mantissa += (1 << nMantBits - 1)
    if scale < largestScale - 1:
        mantissa = ((mantissa << 1) + 1) << (largestScale - scale - 2)

    # if negative flip sign bit
    if sign:
        signBitMask = 1 << R - 1
        mantissa += signBitMask

    aNum = DequantizeUniform(mantissa, R)

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    signBitMask = 1 << nMantBits - 1
    inputVal = aNum

    # Edge Cases
    if nMantBits <= 0:
        return 0.0
    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    if aNum < 0:
        aNum *= -1

    # replace leading zeros
    mantissa = QuantizeUniform(aNum, R) << (scale + 1)
    # shift back
    mantissa >>= R - nMantBits + 1

    if inputVal < 0:
        mantissa += signBitMask

    return mantissa

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    signBitMask = 1 << nMantBits - 1
    sign = 0

    if nMantBits <= 0:
        return 0
    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    if mantissa & signBitMask:
        mantissa -= signBitMask
        sign = 1


    aQuantizedNum = mantissa << largestScale - scale
    # if scale < R and mantissa > 0:
    if scale < largestScale and mantissa > 0:
        aQuantizedNum += 1 << largestScale - scale - 1

    if sign:
        signBitMask = 1 << R - 1
        aQuantizedNum += signBitMask

    aNum = DequantizeUniform(aQuantizedNum, R)

    return aNum

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of
    signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    inputVals = aNumVec.copy() # Copy Values
    signBitMask = 1 << nMantBits - 1

    # Edge Case
    if nMantBits <= 0:
        return zero(N, uint64)
    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    sign = signbit(inputVals)

    inputVals[sign] = -inputVals[sign]

    mantissaVec = vQuantizeUniform(inputVals, R) << (scale + 1)
    mantissaVec >>= R - nMantBits + 1

    mantissaVec[sign] += signBitMask

    return mantissaVec

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and
    vector of block floating-point mantissas given specified scale and mantissa bits
    """

    mantissa = mantissaVec.copy()
    signBitMask = 1 << nMantBits - 1

    # Edge Case
    if nMantBits <= 0:
        return zero(N, uint64)
    if nScaleBits < 0:
        nScaleBits = 0

    largestScale = (1 << nScaleBits) - 1 # (2^Rs - 1)
    R = nMantBits + largestScale # R = (2^Rs - 1) + Rm

    negativeVals = mantissa & signBitMask == signBitMask

    mantissa[negativeVals] -= signBitMask
    aQuantizedNum = mantissa << largestScale - scale

    if scale < largestScale:
        aQuantizedNum[mantissa > 0] += 1 << largestScale - scale - 1

    signBitMask = 1 << R - 1
    aQuantizedNum[negativeVals] += signBitMask

    aNumVec = vDequantizeUniform(aQuantizedNum, R)

    return aNumVec

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    if test_uniform:

        nBits = 8
        Uniform8 = zeros(len(chartInputs), dtype=float64)
        Uniform12 = zeros(len(chartInputs), dtype=float64)
        for i in range(chartInputs.size):
            Uniform8[i] = DequantizeUniform(QuantizeUniform(chartInputs[i], nBits), nBits)
        nBits = 12
        for i in range(chartInputs.size):
            Uniform12[i] = DequantizeUniform(QuantizeUniform(chartInputs[i], nBits), nBits)
        print '\n Uniform8:\n--------------------------\n'
        for i in range(chartInputs.size):
            print '{:06f}'.format(Uniform8[i])
        print '\n Uniform12:\n--------------------------\n'
        for i in range(chartInputs.size):
            print '{:06f}'.format(Uniform12[i])

    if test_uniform_vectorized:

        nBits = 8
        VecUniform8 = vDequantizeUniform(vQuantizeUniform(chartInputs, nBits), nBits)
        print '\n VecUniform8:\n--------------------------\n'
        for i in range(len(chartInputs)):
             print '{:06f}'.format(VecUniform8[i])

        nBits = 12
        VecUniform12 = vDequantizeUniform(vQuantizeUniform(chartInputs, nBits), nBits)
        print '\n VecUniform12:\n--------------------------\n'
        for i in range(len(chartInputs)):
             print '{:06f}'.format(VecUniform12[i])


    if test_fp:

        nScaleBits = 3
        nMantissaBits = 5
        print '\n 3s 5m Floating Point:\n--------------------------\n'
        for input in chartInputs:
            s = ScaleFactor(input, nScaleBits, nMantissaBits)
            m = MantissaFP(input, s, nScaleBits, nMantissaBits)
            print DequantizeFP(s, m, nScaleBits, nMantissaBits)


    if test_bfp:

        nScaleBits = 3
        nMantissaBits = 5
        print '\n 3s 5m Floating Point:\n--------------------------\n'
        # share scaleFactor across block
        s = ScaleFactor(np.max(np.abs(chartInputs)), nScaleBits, nMantissaBits)
        for input in chartInputs:
            m = Mantissa(input, s, nScaleBits, nMantissaBits)
            print Dequantize(s, m, nScaleBits, nMantissaBits)

    if test_bfp_vectorized:

        nScaleBits = 3
        nMantissaBits = 5
        print '\n 3s 5m Floating Point:\n--------------------------\n'

        largestScale = (1 << nScaleBits) - 1 # 2^Rs - 1
        maxMantissa = nMantissaBits + largestScale

        # share scaleFactor across block
        s = ScaleFactor(np.max(np.abs(chartInputs)), nScaleBits, nMantissaBits)
        m = vMantissa(chartInputs, s, nScaleBits, nMantissaBits)
        output = vDequantize(s, m, nScale, nMant)
        for i in range(len(chartInputs)):
            print output[i]

