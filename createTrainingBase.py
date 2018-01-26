"""
Created on Mon Jan 15 14:05:03 2018

@author: Louis BAETENS
"""
import openImages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from extractImagePartPortion import extractRandomRegion
from extractImagePartPortion import extractRandomPortion



def createPylonsBase(numberOfTests, maxDistance):
    '''numberOfTests : nb of image to extract from, maxDistance : 30
    '''
    
    from projectionUMT2pixel import pylonsProjection
    
    
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, "gourd_c1818\\Images\\")
    imPath = os.path.join(currentPath, '..', 'data', 'gourd_c1818', 'Images')
    
#    imPath = "C:\\Users\\Louis\\Google Drive\\PIE Supaero\\PIE_supaero_2017\\data\\gourd_c1818\\Images\\"
    print(imPath)
    
    imagesNames = os.listdir(os.path.join(imPath, '03'))[1:-2]
    
#    numberOfTests = len(imagesNames)
    
#    imagesNumber = np.random.randint(0, len(imagesNames), size=numberOfTests)
    imagesNumber = np.arange(0, numberOfTests, 1)
    
    
    wannaPlot = False
    
    windowSizeX = 100
    windowSizeY = 100
    imageSizeX = 2448
    imageSizeY = 2048
#    maxDistance = 30
    numExtractPerPylon = 1
    
    
    for imNum in imagesNumber:
#        plt.figure()
        imName = imagesNames[imNum]
        print(imName)
        pylonsCoords = pylonsProjection(imPath, imName)
        
        print(pylonsCoords)
        
        pylonsId = [x[0] for x in pylonsCoords if x[1][0]>-1]
        pylonsCoordsX = [x[1][0] for x in pylonsCoords if x[1][0]>-1]
        pylonsCoordsY = [x[1][1] for x in pylonsCoords if x[1][0]>-1]
        
        
        imPath2 = os.path.join(imPath, '03')

        for pylonNum in range(len(pylonsCoordsX)):
            additionalName = '_pyl' +'_{:.0f}'.format(pylonsId[pylonNum])

            for ite in range(numExtractPerPylon):
                try:
                    extractRandomRegion(imPath2, imName, additionalName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, pylonsCoordsX[pylonNum], pylonsCoordsY[pylonNum], maxDistance)
                except:
                    print('An error happened')
        
        if wannaPlot == True:    
            openImages.displayImageMPLT(imPath, imName)

            plt.plot(pylonsCoordsX, pylonsCoordsY, 'or')  


def createLinesBase(numberOfImages, numberOfExtractions, maxDistance):
    '''numberOfImages : nb of image to extract from, numberOfExtractions : 
    nb of extractions per image, maxDistance : 30
    '''
    
    from projectionUMT2pixel import linesProjection
    
    
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, '..', 'data', 'gourd_c1818', 'Images')
    
    print(imPath)
    
    imagesNames = os.listdir(os.path.join(imPath, '03'))[1:-2]
    
    imagesNumber = np.arange(0, numberOfImages, 1)
    
    
    wannaPlot = False
    
    windowSizeX = 100
    windowSizeY = 100
    imageSizeX = 2448
    imageSizeY = 2048
#    maxDistance = 30
    
    
    for imNum in imagesNumber:
#        plt.figure()
        imName = imagesNames[imNum]
        print(imName)
        tempLinesCoords = linesProjection(imPath, imName)
               
        
        tempLinesCoordsX = [x[0] for x in tempLinesCoords if x[0]>-1]
        tempLinesCoordsY = [x[1] for x in tempLinesCoords if x[0]>-1]
        
        if len(tempLinesCoordsX) < 1:
            continue
        
        
        linNumber = np.random.randint(0, len(tempLinesCoordsX), size=numberOfExtractions)     

        linCoordsX = []
        linCoordsY = []
        for lin in linNumber:
            linCoordsX.append(tempLinesCoordsX[lin])
            linCoordsY.append(tempLinesCoordsY[lin])
        
        
        print(linCoordsX)
        print(linCoordsY)        
        
        imPath2 = os.path.join(imPath, '03')
        
        offset = 0 # number to start from for the naming
        for pylonNum in range(numberOfExtractions):
            additionalName = '_lin' +'_{:.0f}'.format(offset+pylonNum)
            try:
                extractRandomRegion(imPath2, imName, additionalName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, linCoordsX[pylonNum], linCoordsY[pylonNum], maxDistance)
            except:
                print('An error happened')  





def createBackgroundBase(numberOfTests, numExtractPerImage):
   
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, '..', 'data', 'gourd_c1818', 'Images')
    
    print(imPath)
    
    imagesNames = os.listdir(os.path.join(imPath, '03'))[1:-2]
    
    imagesNumber = np.arange(0, numberOfTests, 1)
    
    
    wannaPlot = False
    
    windowSizeX = 100
    windowSizeY = 100
    imageSizeX = 2448
    imageSizeY = 2048
#    maxDistance = 30
    
    
    for imNum in imagesNumber:
        imName = imagesNames[imNum]
        print(imName)      
        
        imPath2 = os.path.join(imPath, '03')

        for regionNum in range(numExtractPerImage):
            additionalName = '_bak' +'_{:.0f}'.format(regionNum)

            try:
                extractRandomPortion(imPath2, imName, additionalName, windowSizeX, windowSizeY, imageSizeX, imageSizeY)
#                    extractRandomRegion(imPath2, imName, additionalName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, pylonsCoordsX[pylonNum], pylonsCoordsY[pylonNum], maxDistance)
            except:
                print('An error happened')
        
        if wannaPlot == True:    
            openImages.displayImageMPLT(imPath, imName)

            plt.plot(pylonsCoordsX, pylonsCoordsY, 'or')      

def separateDatabase(dbName, separationType, trainingType):
    '''Separate between a training and a test database, based on various criteria
    SeparationType could be id, imName
    trainingType is a letter, P or L for example
    '''
    from shutil import copyfile

    currentPath = os.path.dirname(os.path.abspath(__file__))
    dbPath = os.path.join(currentPath, '..', 'data_base', dbName+'DB')
    
    nameExtension = trainingType 
    
    subDirectories = ['train'+nameExtension, 'validation'+nameExtension]
    for subDir in subDirectories:
        directory = os.path.join(dbPath, '..', subDir, dbName)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # retrieves all the files names
    patchNames = os.listdir(dbPath)
    print(patchNames)
    
    trainProportion = 0.7
    
    if separationType == 'id':
#        masterList = idList
        nameCut = 3
    elif separationType == 'imName':
#        masterList = originalNames
        nameCut = 1
    
    masterList = [x.split('_')[nameCut] for x in patchNames]
    masterList = sorted(set(masterList))    
    
    
    cutoff = int(np.floor(trainProportion*len(masterList)))
    
    import random
    random.seed(1)  
    
    random.shuffle(masterList)
    
    print(masterList)
    
    trainList = masterList[0:cutoff]
    testList = masterList[cutoff:]

    print(trainList)
    print(testList)

    src = dbPath
    dst = os.path.join(dbPath, '..')    
    for fileName in patchNames:
        if fileName.split('_')[nameCut] in trainList:
            copyfile(os.path.join(src, fileName), os.path.join(dst, 'train'+nameExtension, dbName, fileName))
        elif fileName.split('_')[nameCut] in testList:
            copyfile(os.path.join(src, fileName), os.path.join(dst, 'validation'+nameExtension, dbName, fileName))
        else:
            print(fileName + " dumped")
#            copyfile(os.path.join(src, fileName), os.path.join(dst, dbName+'Dump', fileName))

                
        
    
    
    

def main():

#    createPylonsBase(1865, 20)
#    createBackgroundBase(1865,3)
#    createLinesBase(1865, 5, 1)
    dbName = 'lines'
    separateDatabase(dbName, 'imName', 'L')



if __name__ == '__main__':
    main()