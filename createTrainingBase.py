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

def separateDatabase(dbName, separationType):
    '''Separate between a training and a test database, based on various criteria
    SeparationType could be id, imName
    '''
    from shutil import copyfile


    currentPath = os.path.dirname(os.path.abspath(__file__))
    dbPath = os.path.join(currentPath, '..', 'data_base', dbName+'DB')
    
    subDirectories = ['Train', 'Test']
    for subDir in subDirectories:
        directory = os.path.join(dbPath, '..', dbName+subDir)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # retrieves all the files names
    patchNames = os.listdir(dbPath)
    
    #get the individual original image names and id
    print(patchNames)
#    originalNames = [x.split('_')[1] for x in patchNames]
#    originalNames = sorted(set(originalNames))
#    print(originalNames)
#    
#    idList = [x.split('_')[3] for x in patchNames]
#    idList = sorted(set(idList))
#    print(idList)
    
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
            copyfile(os.path.join(src, fileName), os.path.join(dst, dbName+'Train', fileName))
        elif fileName.split('_')[nameCut] in testList:
            copyfile(os.path.join(src, fileName), os.path.join(dst, dbName+'Test', fileName))
        else:
            print(fileName + " dumped")
#            copyfile(os.path.join(src, fileName), os.path.join(dst, dbName+'Dump', fileName))

                
        
    
    
    

def main():

#    createPylonsBase(1865, 20)
#    createBackgroundBase(1865,3)
    dbName = 'background'
    separateDatabase(dbName, 'imName')



if __name__ == '__main__':
    main()