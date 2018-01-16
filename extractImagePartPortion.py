# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:27:27 2018

@author: Louis BAETENS
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image




def extractPortion(imPath, imName, sizeX, sizeY, centerX, centerY):
    '''Extract a portion of the image of size sizeX x sizeY, with its origin
    being top and left (so top-left corner is to be given)
    Returns the image (or save it, to determine)
    '''
    fullPath = os.path.join(imPath,imName)
    im = Image.open(fullPath)
    
    #set the corners
    left = centerX-sizeX/2
    right = centerX+sizeX/2
    top = centerY-sizeY/2
    bottom = centerY+sizeY/2
    
    crop_rectangle = (left, top, right, bottom)
    
    if left<0 or right>im.size[0] or top<0 or bottom>im.size[1]:
        print("Impossible, bad dimension")
        print(left, top, right, bottom)
        
    else:
        cropped_im = im.crop(crop_rectangle)
        print("Extracted")
        print(left, top, right, bottom)
        outName = (imName[:-4]+'_{:.0f}_{:.0f}.jpg').format(centerX, centerY)
        outPath = os.path.join('croppedImages', outName)

        cropped_im.save(outPath)

#        cropped_im.show()

def extractRandomPortion(imPath, imName, windowSizeX, windowSizeY, imageSizeX, imageSizeY):
    '''Extract a portion of the image, selected randomly
    Not so useful apart for some tests
    '''
    centerX = np.random.randint(windowSizeX/2, imageSizeX-windowSizeX/2)
    centerY = np.random.randint(windowSizeY/2, imageSizeY-windowSizeY/2)    
    extractPortion(imPath, imName, windowSizeX, windowSizeY, centerX, centerY)
    
def extractRandomRegion(imPath, imName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, centerPointX, centerPointY, maxDistance):
    '''Extract a region in the neibourghood of the specified center
    This is the function to use
    '''
    
    #fix the admissible limits, avoid being off the image
    limitLeft = np.max([windowSizeX/2, centerPointX-maxDistance])
    limitTop = np.max([windowSizeY/2, centerPointY-maxDistance])
    limitRight = np.min([imageSizeX-windowSizeX/2, centerPointX+maxDistance])
    limitBottom = np.min([imageSizeY-windowSizeY/2, centerPointY+maxDistance])

    #choose a random center
    if (limitLeft >= limitRight or limitTop >= limitBottom):
        return[centerPointX, centerPointY]
    
    centerX = np.random.randint(limitLeft, limitRight)
    centerY = np.random.randint(limitTop, limitBottom)   
    
    extractPortion(imPath, imName, windowSizeX, windowSizeY, centerX, centerY)
    
    return [centerX, centerY]
    
    
    

def main():
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, 'gourd_c1818', 'Images', '03')
    imName = '03_00000184.jpg'
    
#    sizeX = 500
#    sizeY = 500
#    centerX = 1000
#    centerY = 1000
#    extractPortion(imPath, imName, sizeX, sizeY, centerX, centerY)

    windowSizeX =150
    windowSizeY = 150
    imageSizeX = 2448
    imageSizeY = 2048
    
#    extractPortion(imPath, imName, windowSizeX, windowSizeY, 1000, 800)
    
    
    
    centerX = 1000
    centerY = 800
    maxDistance = 10
    
    centerX = 1000
    centerY = 1000
    maxDistance = 100
    
    allCentersX = []
    allCentersY = []

    for ite in range(50):
#        extractRandomPortion(imPath, imName, windowSizeX, windowSizeY, imageSizeX, imageSizeY)
        centers = extractRandomRegion(imPath, imName, windowSizeX, windowSizeY, imageSizeX, imageSizeY, centerX, centerY, maxDistance)
        allCentersX.append(centers[0])
        allCentersY.append(centers[1])
    
    plt.close('all')
    plt.figure()
    plt.plot(centerX, centerY, 'o')
    plt.plot(allCentersX, allCentersY, 'x')
    
    
if __name__ == '__main__':
    main()