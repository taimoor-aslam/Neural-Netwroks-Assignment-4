# NN 4, part-1
import matplotlib.pyplot as plt
from PIL import Image

def computeY(yin, theta):
    if yin > theta:
        return 1
    elif yin < -theta:
        return -1
    else:
        return 0


def perceptron(samples, targetsVal, weights, b, alpha, theta):
    epoch=1
    errorArray = []
    while True:
        wChange = 0
        j = 0
        errorCount=0
        for sample in samples:
            i = 0
            yin = 0
            for x in sample:
                w = weights[i]
                i = i + 1
                # print(yin)
                yin += x * w
            # print(b)
            yin += b
            y = computeY(yin, theta)
            if y != targetsVal[j]:
                c = 0
                while c < len(weights):
                    weights[c] = weights[c] + alpha * targetsVal[j] * sample[c]  # updating weights
                    wChange = 1
                    c += 1
                errorCount += 1
                b = b + alpha * targetsVal[j]  # updating bias
            j += 1  # next target val
        errorArray.append(errorCount/len(samples)*100)

        if wChange:
            epoch += 1
        else:
            print("Weights are")
            count = 1
#             for i in weights:
#                 print("w" + str(count) + "=" + str(i))
#                 count += 1
            print("Bias is: " + str(b))
            print("Total epochs are: " + str(epoch))
            break
    if wChange!=0:
        break
#      return errorArray
    return weights,b


def toInt(str):
    return list(map(int,str))

def getImage():
    getImage.imageIndex+=1
    if getImage.imageIndex>30:
        return 0
    image=Image.open("E:\\University\\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\Images\\"+str(getImage.imageIndex)+'.png')
    return image

getImage.imageIndex=0


def ImageProcessing(img):
    newSize=(100,100)
    myImg=img.resize(newSize)

    rawData=myImg.load()
    data=[]
    for y in range(100):
        for x in range(100):
            data.append(rawData[x,y])

    pixels=[]
    for val in data:
        str=list(bin(val)[2:].zfill(8))
        pixels.extend(toInt(str))
    return pixels


def TestSample(img,weights,b,theta):
    sample=[]
#     sample.append(ImageProcessing(img))
    sample=ImageProcessing(img)
#     print(len(sample))
    yin = 0
    i=0
    for x in sample:
        w = weights[i]
        i = i + 1
        yin += x * w
    yin += b
    y = computeY(yin, theta)
    if y==1:
        print("Pattren Matched!")
    else:
        print("Pattren not Matched!")
        

def main():
    # Exercise no.1
    b = 0
    alpha = 1
    theta = 0.1
    print("<<<<<Exercise no.1>>>>>")
    samples=[]
    count=0
    while(1):
        img=getImage()
        if(img==0):
            break
        samples.append(ImageProcessing(img))
    print(len(samples))
    
    #for J
    targetsVal=[]
    jTargetFile=open("E:\\University\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\venv\\jTargetSet.txt","r")
    while(1):                            #setting targets values
        mystr=jTargetFile.readline()
        if not mystr:
            break
        targetsVal.append(mystr[:-1])    
    weights=[0]*80000
    targetsVal=toInt(targetsVal)
    calWeights,calBiase=perceptron(samples,targetsVal,weights,b,alpha,theta)
    Jimage=Image.open("E:\\University\\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\Images\\9.png")
    TestSample(Jimage,calWeights,calBiase,theta)
    
    #for K
    targetsVal=[]
    kTargetFile=open("E:\\University\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\venv\\kTargetSet.txt","r")
    while(1):                            #setting targets values
        mystr=kTargetFile.readline()
        if not mystr:
            break
        targetsVal.append(mystr[:-1])    
    weights=[0]*80000
    targetsVal=toInt(targetsVal)
    calWeights,calBiase=perceptron(samples,targetsVal,weights,b,alpha,theta)
    kimage=Image.open("E:\\University\\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\Images\\15.png")
    TestSample(kimage,calWeights,calBiase,theta)
    
    #for L
    targetsVal=[]
    lTargetFile=open("E:\\University\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\venv\\lTargetSet.txt","r")
    while(1):                            #setting targets values
        mystr=lTargetFile.readline()
        if not mystr:
            break
        targetsVal.append(mystr[:-1])    
    weights=[0]*80000
    targetsVal=toInt(targetsVal)
    calWeights,calBiase=perceptron(samples,targetsVal,weights,b,alpha,theta)
    Limage=Image.open("E:\\University\\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\Images\\25.png")
    TestSample(Limage,calWeights,calBiase,theta)
    

main()


