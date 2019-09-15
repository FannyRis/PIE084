"""
Created on Tue Jan  9 14:07:53 2018

@author: Louis BAETENS
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

   
def displayImageMPLT(imPath, imName):
    imPath = os.path.join(imPath, "03", imName)
    img = mpimg.imread(imPath)
    plt.imshow(img)
    
def drawPointOnImage(imPath, imName, pX, pY):
    displayImageMPLT(imPath, imName)
    plt.scatter(pX, pY)


def getPhotoParameters(imPath, imName, cameraParamFile):
    """Return the photo parameters (X,Y,Z,omega, phi, kappa) given its name
    """

    cameraParamPath = os.path.join(imPath, cameraParamFile)

    file = open(cameraParamPath, 'r')
    lines = file.readlines()
    file.close()

    for line in lines:
        imagesParameters = line.split(" ")
        if imagesParameters[0] == imName:
            break

    imagesParameters[1:] = [float(i) for i in imagesParameters[1:]]

    return imagesParameters

def getCameraParameters(imPath):
    '''Get the internal camera parameters and return them
    '''
    
    cameraInternalFile = "03_calibrated_internal_camera_parameters.cam"
    cameraInternalParamPath = os.path.join(imPath, cameraInternalFile)
    file = open(cameraInternalParamPath, 'r') 
    lines = file.readlines()
    file.close()
    
    captorSizeX = lines[1].split("a sensor width of ")[1].split("x")[0]
    captorSizeY = lines[1].split("a sensor width of ")[1].split("x")[1].replace("mm", "") 
    focal = lines[2].split(" ")[1]
    Px = lines[4].split(" ")[1]
    Py = lines[5].split(" ")[1]
    K1 =  lines[7].split(" ")[1]
    K2 =  lines[8].split(" ")[1]    
    K3 =  lines[9].split(" ")[1]
    T1 =  lines[11].split(" ")[1]
    T2 =  lines[12].split(" ")[1]
    
    internalParameters = (captorSizeX, captorSizeY, focal, Px, Py, K1, K2, K3, T1, T2)
    
    #transform them into float
    internalParameters = [float(i) for i in internalParameters]
#    print(internalParameters)
    return internalParameters


def getDistance2Camera(pylonCoords, cameraCoords):
    '''Gives the distance between a pylon and the camera in X and Y
    '''
    
    pylonCoords = [float(i) for i in pylonCoords]   
    cameraCoords = [float(i) for i in cameraCoords]   

    
    vector = [cameraCoords[0]-pylonCoords[0], cameraCoords[1]-pylonCoords[1]]
    distance = np.linalg.norm(vector)
    return distance 


def getPotentialPylones(imPath, imName, maxDistanceToCamera, cameraParamFile):
    ''' Returns the IDs and coordinates of the potential pylones on the image, given a constraint
    on their admissible distance to the camera
    '''
    
    photoParameters = getPhotoParameters(imPath, imName, cameraParamFile)
    cameraPos = photoParameters[1:3]
    
#    maxDistanceToCamera = 100
    
    potentialPylones = []
    
    with open('pylonsGPS.csv', 'r') as csvfile:
        pylonsList = csv.reader(csvfile)
        pylonsList = filter(None, pylonsList)
        for pylon in pylonsList:
            pylon = [float(i) for i in pylon]
            distance = getDistance2Camera(pylon[1:3], cameraPos)
            if distance<maxDistanceToCamera:
                potentialPylones.append(pylon)

    
    return potentialPylones

def getPotentialLinePoints(imPath, imName, maxDistanceToCamera, cameraParamFile):
    ''' Returns the IDs and coordinates of the potential lines on the image, given a constraint
    on their admissible distance to the camera
    '''
    
    photoParameters = getPhotoParameters(imPath, imName, cameraParamFile)
    cameraPos = photoParameters[1:3]
    
    potentialLines = []

    
    with open('all_linesGPS_center.csv', 'r') as csvfile:
        linesList = csv.reader(csvfile)
        linesList = filter(None, linesList)
        for pylon in linesList:
            pylon = [float(i) for i in pylon]
            distance = getDistance2Camera(pylon[0:2], cameraPos)
            if distance<maxDistanceToCamera:
                potentialLines.append(pylon)
                
    return potentialLines



def main():
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, "gourd_c1818\\Images\\")
    print(imPath)
    imName = "03_00000175.jpg"
    
#    drawPointOnImage(imPath, imName, 100, 400)
#
#    displayImageMPLT(imPath, imName)
    
    linesPoints = getPotentialLinePoints(imPath, imName, 300)
    print(linesPoints)
    
#    displayImage(imPath, imName)
#    getPhotoParameters(imPath, imName)
#    getCameraParameters(imPath)
#    potentials = getPotentialPylones(imPath, imName, 500)
#    pprint(potentials)

if __name__ == '__main__':
    main()
#    cProfile.run('main()', 'tgz.txt')
#    import pstats
#    p = pstats.Stats('tgz.txt')
#    p.sort_stats('cumulative').print_stats(10)
    
#    p.strip_dirs().sort_stats(-1).print_stats()