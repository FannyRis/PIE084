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

def createPylonsBase():
    
    from projectionUMT2pixel import pylonsProjection
    
    
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, "gourd_c1818\\Images\\")
    
    imPath = "C:\\Users\\Louis\\Google Drive\\PIE Supaero\\PIE_supaero_2017\\data\\gourd_c1818\\Images\\"
    print(imPath)
    
    imagesNames = os.listdir(os.path.join(imPath, '03'))[1:-2]
    
    numberOfTests = len(imagesNames)
    
    imagesNumber = np.random.randint(0, len(imagesNames), size=numberOfTests)
    
#    imagesNumber = [1104]

    
    wannaPlot = False
    
    windowSizeX = 100
    windowSizeY = 100
    imageSizeX = 2448
    imageSizeY = 2048
    maxDistance = 30
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
            for ite in range(numExtractPerPylon):
                extractRandomRegion(imPath2, imName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, pylonsCoordsX[pylonNum], pylonsCoordsY[pylonNum], maxDistance)
        
        
        if wannaPlot == True:    
            openImages.displayImageMPLT(imPath, imName)

            plt.plot(pylonsCoordsX, pylonsCoordsY, 'or')  



def main():
    createPylonsBase()



if __name__ == '__main__':
    main()