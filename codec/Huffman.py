# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import deque
import cPickle as pickle

#------------------------------------------------------------------------#
# Created by: Kai-Chieh Huang // Feb, 21, 2016 // kaichieh@stanford.edu  #
#------------------------------------------------------------------------#
'''
This file implements a Huffman encoding class to compress audio files
'''
#------------------------------------------------------------------------#
class HuffmanNode:
    freq = None
    mantissaCode = None
    zero = None #left Node
    one = None  #right Node
    def __init__(self,mantissaCode=None,freq=None,left=None,right=None):
        self.zero = left
        self.one = right
        self.freq = freq
        self.mantissaCode = mantissaCode

#------------------------------------------------------------------------#
class Histogram:
    
    # Stores the occurence probability of each symbol
    probability = dict()
    # Stores the occurence frequency of each symbol
    statistics = dict()
    # Stores the Huffman Nodes to be pop out (prioritized based on freq)
    queue = deque()
    
    #---PUBLIC METHODS---#
    def __init__(self):
        self.LOW_FREQ = 10
        self.ESCAPE_CODE = -1
        pass

    #----------------------------------------------------------#
    #. getMatchScore(blockHistogram)
    #. This function takes in the histogram of the block and see
    #. how good it matches this histogram and return the match
    #. score.
    #. @Param:  the histogram object represent the block
    #. @Return: score
    #----------------------------------------------------------#
    def getMatchScore(self,blockHistogram):
        # use probability to compare and give score
        histogramProb = self.probability
        # Sort the HistogramProb based on the mantissa code value for alignment
        sortedHistogramProb = OrderedDict(sorted(histogramProb.items(), key=lambda t: t[0]))
        sortedHistogramProb = np.array(sortedHistogramProb.values())
         # Sort the block HistogramProb based on the mantissa code value for alignment
        sortedBlockHistogramProb = OrderedDict(sorted(blockHistogram.probability.items(),key=lambda t: t[0]))
        sortedBlockHistogramProb = np.array(sortedBlockHistogramProb.values())
        #print "Block prob: ",sortedBlockHistogramProb
        difference = sum((sortedHistogramProb - sortedBlockHistogramProb)**2)
        return (3.0-difference)
        
    #----------------------------------------------------------#
    #. generateStatistics(mantissaCode)
    #. This function takes in a list of mantissa code (*without
    #. sign bit*) and count the frequency of the occurence of
    #. paricular mantissa code to generate the statistics.
    #. @Param:  a list of mantissa code (without sign bit)
    #. @Return: void
    #----------------------------------------------------------#
    def generateStatistics(self,mantissaCode):
        # Count the occurence freq for each manitssaCode
        for i in range(len(mantissaCode)):
            if mantissaCode[i] in self.statistics:
                self.statistics[mantissaCode[i]] = self.statistics[mantissaCode[i]] + 1
            else:
                self.statistics[mantissaCode[i]] = 1
        totalCount = sum(self.statistics.values())
        # Transform statistics to probabilities for finding histogram matches
        for key, value in self.statistics.items():
            self.probability[key] = value/float(totalCount)

    #------------------------------------------------------------#
    #. makeHuffmanNodeQueue()
    #. This function uses the statistics inside this Histogram
    #. object to generate a prioritized queue based on the freq
    #. of the occurence of each symbol, should call this function
    #. before calling getNextPair().
    #. @Param:  void
    #. @Return: void
    #------------------------------------------------------------#
    def makeHuffmanNodeQueue(self):
        # sort the statistics from low freq to high freq
        sortedStatistics = OrderedDict(sorted(self.statistics.items(), key=lambda t: t[1]))
        sortedStatistics = sortedStatistics.items()
        escapeFreq = 0
        for pair in sortedStatistics:
            # pair[0] -> mantissa code
            # pair[1] -> frequency
            # Skip very low frequency symbols and replace with escape code
            if pair[1] < self.LOW_FREQ:
                escapeFreq += 1
            else:
                newNode = HuffmanNode(pair[0],pair[1])
                self.queue.append(newNode)
        # Add the node for escape code
        newNode = HuffmanNode(self.ESCAPE_CODE ,escapeFreq)
        self.queue.append(newNode)
        self.queue = deque(sorted(self.queue, key=lambda t: t.freq))

    #------------------------------------------------------------#
    #. appendToHuffmanQueue(huffmanNode)
    #. The Histogram class also serve as the function of priority
    #. queue in Huffman encoding to enqueue the joint Huffman node.
    #. @Param:  Huffman node (joined huffman node)
    #. @Return: void
    #------------------------------------------------------------#
    def appendToHuffmanQueue(self,huffmanNode):
        self.queue.append(huffmanNode)
        self.queue = deque(sorted(self.queue, key=lambda t: t.freq))

    #------------------------------------------------------------#
    #. getNextPair()
    #. The Histogram class also serve as the function of priority
    #. queue in Huffman encoding to pop 2 lowest freq symbols each
    #. time.
    #. @Param:  void
    #. @Return: a pair of Huffman Node
    #------------------------------------------------------------#
    def getNextPair(self):
        if len(self.queue)==1:
            return (self.queue.popleft(),None)
        firstNode = self.queue.popleft()
        secondNode = self.queue.popleft()
        return (firstNode,secondNode)

#------------------------------------------------------------------------#
class HuffmanTable:
    
    #---PUBLIC METHODS---#
    #----------------------------------------------------------#
    #. HuffmanTable constructor
    #. This is a convenient class to encapsulate the huffman
    #. table. It supports the two way huffman table for encoding
    #. and decoding.
    #. @Param:  a dictionary of {manitssaCode : HuffmanCode}
    #. @Return: void
    #----------------------------------------------------------#
    def __init__(self,manitssaCodeToHuffmanCode):
        self.encodingTable = manitssaCodeToHuffmanCode
        self.decodingTable = dict()
        for key, value in manitssaCodeToHuffmanCode.items():
            self.decodingTable[value] = key

#------------------------------------------------------------------------#
class HuffmanTrainer:
    
    histogram = Histogram()
    huffmanCodeTable = dict()
    root = HuffmanNode()
    
    #---PUBLIC METHODS---#
    #----------------------------------------------------------#
    #. HuffmanTrainer constructor
    #. This function takes in the ID of the Huffman table to be
    #. constructed and then init the HuffmanTrainer object.
    #. @Param:  table ID
    #. @Return: void
    #----------------------------------------------------------#
    def __init__(self,tableID):
        self.tableID = tableID
        self.histogram = Histogram()

    #----------------------------------------------------------#
    #. countFreq(mantissaCode)
    #. This function takes in a list of mantissa code (*without
    #. sign bit*) and count the frequency of the occurence of
    #. paricular mantissa code.
    #. @Param:  a list of mantissa code (without sign bit)
    #. @Return: void
    #----------------------------------------------------------#
    def countFreq(self,mantissaCode):
        self.histogram.generateStatistics(mantissaCode)
    
    #----------------------------------------------------------#
    #. constructHuffmanTable()
    #. This function construct a Huffman code table based on the
    #. statistics it collects so far and store the table with
    #. corresponding ID in order for future decode lookup.
    #. @Param:  void
    #. @Return: void
    #----------------------------------------------------------#
    def constructHuffmanTable(self):
        self.histogram.makeHuffmanNodeQueue()
        self.__buildEncodingTree()
        self.__buildEncodingTable()
        # Save the huffman code table to file to read in for decode
        with open('huffmanTables.pickle', 'rb') as handle:
            huffmanTables = pickle.load(handle)
            huffmanTables[self.tableID] = HuffmanTable(self.huffmanCodeTable)
            with open('huffmanTables.pickle', 'wb') as handle:
                pickle.dump(huffmanTables, handle)
        # Save the corresponding histogram to file to read in for decode
        with open('histograms.pickle', 'rb') as handle:
            histograms = pickle.load(handle)
            histograms[self.tableID] = self.histogram
            with open('histograms.pickle', 'wb') as handle:
                pickle.dump(histograms, handle)

    #---PRIVATE METHODS---#
    #----------------------------------------------------------#
    #. __buildEncodingTree()
    #. Private method to build the huffman encoding tree and sets
    #. the root huffman node for building encoding table.
    #. @Param:  void
    #. @Return: void
    #----------------------------------------------------------#
    def __buildEncodingTree(self):
        while True:
            (firstNode,secondNode) = self.histogram.getNextPair()
            if secondNode == None:
                self.root = firstNode
                break
            joinedNode = HuffmanNode(None,firstNode.freq+secondNode.freq,firstNode,secondNode)
            self.histogram.appendToHuffmanQueue(joinedNode)

    #----------------------------------------------------------#
    #. __buildEncodingTableHelper()
    #. Private recursive helper method to build the huffman
    #. encoding table.
    #. @Param:  void
    #. @Return: void
    #----------------------------------------------------------#
    def __buildEncodingTableHelper(self,node,huffmanCode):
        if node.mantissaCode != None:
            # Need to transform huffman code from string to int
            self.huffmanCodeTable[node.mantissaCode] = huffmanCode
            return
        self.__buildEncodingTableHelper(node.zero,huffmanCode+"0")
        self.__buildEncodingTableHelper(node.one,huffmanCode+"1")

    #----------------------------------------------------------#
    #. __buildEncodingTable()
    #. Private method to build the huffman encoding table.
    #. @Param:  void
    #. @Return: void
    #----------------------------------------------------------#
    def __buildEncodingTable(self):
        huffmanCode = ""
        self.__buildEncodingTableHelper(self.root,huffmanCode)

#------------------------------------------------------------------------#
class Huffman:
    
    #---PUBLIC METHODS---#
    def __init__(self):
        with open('huffmanTables.pickle', 'rb') as handle:
            self.huffmanTables = pickle.load(handle)
        with open('histograms.pickle', 'rb') as handle:
            self.histograms = pickle.load(handle)
        self.ESCAPE_CODE = -1
        self.bitDeposit = 0
    
    #----------------------------------------------------------#
    #. encodeData(mantissaCode)
    #. This function takes in a list of mantissa code (*without
    #. sign bit*) and maps them into Huffman codes.
    #. @Param:  coding parameters
    #. @Param:  a list of mantissa code (without sign bit)
    #. @Param:  a bitAlloc list information for doing escape code
    #. @Return: a list of huffman code, and the tableID it used
    #.          to encode the mantissa codes.
    #----------------------------------------------------------#
    def encodeData(self,codingParams,mantissaCode,bitAlloc):
        bestHuffmanCodedMantissa = []
        #totalBitLength = sum(bitAlloc*codingParams.sfBands.nLines)
        bestTableID = 1
        # construct histogram for this block
        #blockHistogram = Histogram()
        #blockHistogram.generateStatistics(mantissaCode)
        # find the best match huffman table
        #bestMatchTableID = self.__findBestMatchHuffmanTable(blockHistogram)
        # encode the shit
        for ID in self.huffmanTables.keys():
            huffmanCodedMantissa = []
            huffmanTable = self.huffmanTables[ID]
            iMant=0
            for iBand in range(codingParams.sfBands.nBands):
                if bitAlloc[iBand]:
                    for j in range(codingParams.sfBands.nLines[iBand]):
                        code = mantissaCode[iMant+j]
                        if code in huffmanTable.encodingTable.keys():
                            huffmanCodedMantissa.append(huffmanTable.encodingTable[code])
                        # If mantissa code not presented in table, use escape code and code the origin mantissa
                        else:
                            form = '0' + str(bitAlloc[iBand]) + 'b'
                            mantString = format(code,form)
                            huffmanCodedMantissa.append(huffmanTable.encodingTable[self.ESCAPE_CODE]+mantString)
                    iMant += codingParams.sfBands.nLines[iBand]
            totalHuffmanBitLength = sum(len(huff) for huff in huffmanCodedMantissa)
            if  ID == 1:
                totalBitLength = totalHuffmanBitLength
                bestHuffmanCodedMantissa = huffmanCodedMantissa
                bestTableID = ID
            if  totalHuffmanBitLength < totalBitLength:
                totalBitLength = totalHuffmanBitLength
                bestHuffmanCodedMantissa = huffmanCodedMantissa
                bestTableID = ID
        return (bestHuffmanCodedMantissa,bestTableID)
    
    #----------------------------------------------------------#
    #. decodeData(huffmanCode,tableID)
    #. This function takes in a bit reader and the table ID used
    #. to generate this huffman code, and use the Huffman tree
    #. to find corresponding mantissa codes (without sign bit.)
    #. @Param:  a bit reader, and tableID to use for decode.
    #. @Param:  a bitAlloc information for recover mantissa from
    #.          escape code.
    #. @Return: a mantissa code (without sign bit)
    #----------------------------------------------------------#
    def decodeData(self,bitReader,tableID,bitAlloc):
        huffmanTable = self.huffmanTables[tableID]
        huffmanCode = ""
        huffmanCodes = huffmanTable.decodingTable.keys()
        mantissa = self.decodeDataHelper(bitReader,huffmanTable,huffmanCodes,huffmanCode)
        if mantissa == self.ESCAPE_CODE:
            return bitReader.ReadBits(bitAlloc)
        return mantissa
    
    #----------------------------------------------------------#
    #. decodeDataHelper(bitReader,rootNode)
    #. This function takes in a bit reader and the Huffman table
    #. used to encode the manitssa and decode the Huffman code.
    #. @Param:  a bit reader, a huffman table
    #. @Return: the mantissa code recovered (without sign bit)
    #----------------------------------------------------------#
    def decodeDataHelper(self,bitReader,huffmanTable,huffmanCodes,huffmanCode):
        while True:
            if huffmanCode in huffmanCodes:
                return huffmanTable.decodingTable[huffmanCode]
            if(bitReader.ReadBits(1)==0):
                huffmanCode += repr(0)
            else:
                huffmanCode += repr(1)

    #----------------------------------------------------------#
    #. depositBits(numBits)
    #. This function takes in number of bits to be deposit for
    #. bit re-allocation.
    #. @Param:  a number of bits to deposit
    #. @Return: void
    #----------------------------------------------------------#
    def depositBits(self,numBits):
        self.bitDeposit += numBits

    #----------------------------------------------------------#
    #. withdrawBits()
    #. This function takes withdraws 1% of the bit deposit for
    #. bit re-allocation.
    #. @Param:  void
    #. @Return: a number of bits
    #----------------------------------------------------------#
    def withdrawBits(self):
        extraBit = 0
        if self.bitDeposit > 10:
            extraBit = self.bitDeposit/100
            self.bitDeposit -= extraBit
        elif self.bitDeposit < 0:
            extraBit = self.bitDeposit
            self.bitDeposit = 0
        return extraBit

    def getBitDeposit(self):
        return self.bitDeposit

    #---PRIVATE METHODS---#
    #----------------------------------------------------------#
    #. __findBestMatchHuffmanTable(blockHistogram)
    #. This function takes the histogram of the block to be
    #. encode and returns the huffman table ID of the best match.
    #. @Param:  histogram of the block to be encode
    #. @Return: the best match huffman table ID
    #----------------------------------------------------------#
#    def __findBestMatchHuffmanTable(self,blockHistogram):
#        bestScore = 0
#        bestMatchTableID = 1
#        for ID, histogram in self.histograms.items():
#            if histogram.getMatchScore(blockHistogram) > bestScore:
#                bestMatchTableID = ID
#        return bestMatchTableID

#------------------------------------------------------------------------#
"""---TESTING!----"""
if __name__ == "__main__":
    
    with open('huffmanTables.pickle', 'rb') as handle:
        huffmanTables = pickle.load(handle)
        for id in huffmanTables.keys():
            print "Table ",id,":", huffmanTables[id].encodingTable[-1]

#    with open('huffmanTables.pickle', 'wb') as handle:
#        pickle.dump(dict(), handle)
#    with open('histograms.pickle', 'wb') as handle:
#        pickle.dump(dict(), handle)
#    with open('huffmanTrees.pickle', 'wb') as handle:
#        pickle.dump(dict(), handle)

#    trainer = HuffmanTrainer(1)
#    mantissaCode1 = [13,11,13,11,24,11,25]
#    trainer.countFreq(mantissaCode1)
#    trainer.constructHuffmanTable()
#    huffman = Huffman()
#    huffmanCode = huffman.encodeData(mantissaCode1)
#    print huffmanCode
#    print huffman.decodeData(huffmanCode,1)
#    mantissaCode2 = [0,0,1,1,2,2,3,4,4,4]
#    trainer = HuffmanTrainer(1)
#    trainer.countFreq(mantissaCode2)
#    trainer.constructHuffmanTable()
#    huffman = Huffman()
#    huffmanCode = huffman.encodeData(mantissaCode2)
#    print huffmanCode
    pass
