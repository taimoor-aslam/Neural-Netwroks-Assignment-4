
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
    while epoch <= 100:
        wChange = 0
        j = 0
        errorCount=0
        i=0
        for sample in samples:
            yin = [0,0,0]
            while(i<len(weights)):   
                for x in sample:
                    weight = weights[i]
                    for w in weight:
                        yin[i] += x * w
                    # print(yin)       
                # print(b)
                yin[i] += b[i]
                y[i] = computeY(yin[i], theta)
                i+=1
            if y != targetsVal[j]:
                count = 0
                column=0
                for weight in weights:
                    for w in weight:
                        weights[count][column] = w + alpha * targetsVal[j][count] * sample[count]  # updating weights
                        wChange = 1
                        column+=1
                    errorCount += 1
                    b[count] = b[count] + alpha * targetsVal[count]  # updating bias
                    count += 1
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
#      return errorArray
    return weights,b


def toIntArray(str):
    intArray=[]
    for val in str:
        obj=val.split(" ")
#         print(obj)
        intArray.append(list(map(int,obj)))
#     print(intArray)
    return intArray

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
#             print(data)

    pixels=[]
    for val in data:
        str=list(bin(val)[2:].zfill(8))
        pixels.extend(toInt(str))
#         print(pixels)
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
    b = [0,0,0]
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
#     print(len(samples))
    
    targetsVal=[]
    TargetFile=open("E:\\University\BOOKS\\7th Semester\\NN(Neural Networks)\\A4\\venv\\jklTargetSet.txt","r")
    while(1):                            #setting targets values
        mystr=TargetFile.readline()
        if not mystr:
            break
        targetsVal.append(mystr[:-1])   
    weights=[[0]*80000,[0]*80000,[0]*80000]
#     weights[0]=[0]*80000
#     weights[1]=[0]*80000
#     weights[2]=[0]*80000
#     print(targetsVal)
    targetsVal=toIntArray(targetsVal)
#     print(targetsVal)
    calWeights,calBiase=perceptron(samples,targetsVal,weights,b,alpha,theta)
    

main()
