#!/usr/bin/python
# coding=utf-8
"""
description:
    The implementation of calculating MRR/MAP/ACC@1
    Usage: 'python evaluation.py QApairFile scoreFile outputFile'

author:    yafuilee

Created on 2017.7.26

"""
import sys
import codecs


class Evaluator(object):
    qIndex2aIndex2aScore = {}
    qIndex2aIndex2aLabel = {}
    ACC_at1List = []
    APlist = []
    RRlist = []

    def __init__(self, qaPairFile, scoreFile):
        self.loadData(qaPairFile, scoreFile)

    def loadData(self, qaPairFile, scoreFile):
        qaPairLines = codecs.open(qaPairFile, 'r', 'utf-8').readlines()
        scoreLines = open(scoreFile).readlines()
        assert len(qaPairLines) == len(scoreLines)
        qIndex = 0
        aIndex = 0
        label = 0
        score = 0.0
        lastQuestion = ''
        question = ''
        #         answer=''
        for idx in range(len(qaPairLines)):
            qaLine = qaPairLines[idx].strip()
            qaLineArr = qaLine.split('\t')
            assert len(qaLineArr) == 3
            question = qaLineArr[0]
            #             answer=qaLineArr[1]
            label = int(qaLineArr[2])
            score = float(scoreLines[idx])
            if question != lastQuestion:
                if idx != 0:
                    qIndex += 1
                aIndex = 0
                lastQuestion = question
            if not qIndex in self.qIndex2aIndex2aScore:
                self.qIndex2aIndex2aScore[qIndex] = {}
                self.qIndex2aIndex2aLabel[qIndex] = {}
            self.qIndex2aIndex2aLabel[qIndex][aIndex] = label
            self.qIndex2aIndex2aScore[qIndex][aIndex] = score
            aIndex += 1

    def calculate(self):
        for qIndex, index2scoreList in self.qIndex2aIndex2aScore.items():
            index2label = self.qIndex2aIndex2aLabel[qIndex]

            rankIndex = 0
            rightNum = 0
            curPList = []
            rankedList = sorted(index2scoreList.items(), key=lambda b: b[1], reverse=True)
            self.ACC_at1List.append(0)
            for info in rankedList:
                aIndex = info[0]
                label = index2label[aIndex]
                rankIndex += 1
                if label == 1:
                    rightNum += 1
                    if rankIndex == 1:
                        self.ACC_at1List[-1] = 1
                    p = float(rightNum) / rankIndex
                    curPList.append(p)
            if len(curPList) > 0 and len(curPList) != len(rankedList):
                self.RRlist.append(curPList[0])
                self.APlist.append(float(sum(curPList)) / len(curPList))
            else:
                self.ACC_at1List.pop()

    def MRR(self):
        return float(sum(self.RRlist)) / len(self.RRlist)

    def MAP(self):
        return float(sum(self.APlist)) / len(self.APlist)

    def ACC_at_1(self):
        return float(sum(self.ACC_at1List)) / len(self.ACC_at1List)


def evaluate(QApairFile, scoreFile, outputFile='evaluation.score'):
    testor = Evaluator(QApairFile, scoreFile)
    testor.calculate()
    print("MRR:%f \t MAP:%f \t ACC@1:%f\n" % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))
    if outputFile != '':
        fw = open(outputFile, 'a')
        fw.write('%f \t %f \t %f\n' % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))


if __name__ == '__main__':
    #     QApairFile='testing.data'
    #     scoreFile='predictRst.score'
    #     outputFile = ''
    QApairFile = sys.argv[1]
    scoreFile = sys.argv[2]
    outputFile = sys.argv[3]
    evaluate(QApairFile, scoreFile, outputFile)
