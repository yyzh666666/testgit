#/usr/bin/env python


import sys

import math

import pdb


TrainData = []

LMD = 0.5 #lambda, step size

MinLMD = 1.0e-30
ParamNum = 5 #number of parameters
Betas = [0.0,0.0,0.0,0.0,0.0]
NewBetas = [0.5,0.5,0.5,0.5,0.5]
ALPHA = 0.4



#read training data
def ReadData(trainingDataFile):
        infile = file(trainingDataFile, 'r')
        sline = infile.readline().strip()
        while len(sline) > 0:
                fields = sline.strip().split(" ")

                fvector = []

                for field in fields:

                        fvector.append(float(field))

                #append fake feature for beta0
                fvector.append(1.0)

                #used for computing exp(Z) (Z is the dot product of feature weight vector and example vector)
                fvector.append(0.0)

                #print vector

                TrainData.append(fvector)
                sline = infile.readline().strip()
        infile.close()
        #for eachrow in TrainData:

        #       for eachfield in eachrow:

        #               print eachfield,
        #       print
        print len(TrainData),"lines loaded!"

#compute exp(Z)(Z is the dot product of feature weight vector and example vector)


def ExpZ(fv, weis):

        f = 0.0

        for i in range(len(weis)):

                f += weis[i]*fv[i]

        return math.exp(f)

def Predict(v,w):
        f = ExpZ(v,w)
        return f/(float)(1+f)

def Mode(v):

        sum = 0.0

        for f in v:

                sum += f*f

        sum = math.sqrt(sum)

        return sum



#compute the negative log-likelihood of the whole training data
def ComputeLL(weis):

        ll = 0.0

        f = 0.0

        for anexample in TrainData:

                f = ExpZ(anexample[2:],weis)

                ll += anexample[0]*math.log(f/(float)(1+f))

                ll -= anexample[1]*math.log(1+f)

        return -ll



def Iterate():

        #pdb.set_trace()

        global LMD
        f = 0.0

        ll = 0.0

        i = 0

        for anexample in TrainData:

                f = ExpZ(anexample[2:],Betas)

                TrainData[i][7] = f

                i += 1

                ll += anexample[0]*math.log(f/(float)(1+f))

                ll -= anexample[1]*math.log(1+f)

        ll = -ll
        print ll
        print Betas
        wv = []

        for i in range(ParamNum):

                sum = 0.0

                for anexample in TrainData:
                        f = anexample[7]
                        newf = anexample[1]*f-anexample[0]

                        newf *= 1/(float)(1+f)

                        sum += newf*anexample[i+2]

                #print sum

                wv.append(sum)

        mode  = Mode(wv)

        newll = ll

        LMD = 2*LMD

        #pdb.set_trace()

        while ll <= (newll + ALPHA * LMD * mode) and LMD > MinLMD:

                LMD /= 2

                #print LMD
                #print ll
                #print newll

                #pdb.set_trace()

                for i in range(0, ParamNum):

                        NewBetas[i] = Betas[i]-LMD*wv[i]/mode

                newll = ComputeLL(NewBetas)

        for i in range(0,ParamNum):

                Betas[i] = NewBetas[i]



trainingFile = "D:\\LR\\LRTrainNew.txt"


ReadData(trainingFile)
itnum = 0

while itnum < 500 and LMD > MinLMD:
 
        itnum += 1

        print itnum

        Iterate()

print Betas

for anexample in TrainData:
        #pdb.set_trace()
        p = Predict(anexample[2:7], Betas)
        print p
