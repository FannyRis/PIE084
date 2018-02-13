
import openImages
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import os
import string
import csv

# ______________________________________________________________________________________________________________________
# ======================================================================================================================
#                                                V A R I A B L E S

cameraParamFile = "03_calibrated_external_camera_parameters_UTM31.txt"

# ______________________________________________________________________________________________________________________
# ======================================================================================================================
#                                               M E T H O D S

def m2mm(l):
    """Convert meters to millimeters """
    return l / 1000

# **********************************************************************************************************************

def calculateImageParameters(imagePath, imageName):
    """Calculate the Image parameters from the camera and image data
    Allows to compute multiple projections on one image faster
    """
    
    (imageName, xCamera, yCamera, zCamera, omega, phi, kappa) = openImages.getPhotoParameters(imagePath, imageName, cameraParamFile)
    (captorSizeX, captorSizeY, focal, px, py, K1, K2, K3, T1, T2) = openImages.getCameraParameters(imagePath)
    
    img = mpimg.imread(os.path.join(imagePath, '03', imageName))
    (imageSizeY,imageSizeX,d) = img.shape
   
    captorSizeX = m2mm(captorSizeX)
    captorSizeY = m2mm(captorSizeY)
    f = m2mm(focal)
    px = m2mm(px)
    py = m2mm(py)
    
    coordCamera = [xCamera, yCamera, zCamera]
    
    omega = np.deg2rad(omega)
    phi = np.deg2rad(phi)
    kappa = np.deg2rad(kappa)
    
    rKappa = [[cos(kappa), sin(kappa), 0], [-sin(kappa), cos(kappa), 0], [0, 0, 1]]
    rOmega = [[1,0,0], [0, cos(omega), sin(omega)], [0, -sin(omega), cos(omega)]]
    rPhi = [[cos(phi), 0, -sin(phi),], [0, 1, 0], [sin(phi), 0, cos(phi)]]
    
    r = np.dot(np.dot(rKappa,rPhi),rOmega)
           
    px_pixel = px*imageSizeX/captorSizeX
    py_pixel = py*imageSizeY/captorSizeY
    K = [[f, 0, px],
         [0, f, py],
         [0, 0, 1]]
    
    return r, coordCamera, K, imageSizeX, imageSizeY, captorSizeX, captorSizeY, px, py, px_pixel, py_pixel, f, K1, K2, K3, T1, T2

# **********************************************************************************************************************

def projUMT2pixel(PointUMTCoordinates, imageParameters):
    '''Project a UMT31 point on the image, given its parameters
    and those of the camera. Returns the pixel coordinates of the projected point
    imageParameters are
    (R, C, K, imageSizeX, imageSizeY, captorSizeX, captorSizeY, px, py, px_pixel, py_pixel, f, K1, K2, K3, T1, T2),
    computed before from the function calculateImageParameters
    '''
    
    (R, cameraUMTCoordinates, K, imageSizeX, imageSizeY, captorSizeX, captorSizeY, px, py, px_pixel, py_pixel, f, K1, K2, K3, T1, T2) = imageParameters
    
    pointCameraCoordinates =  np.dot(R, np.subtract(PointUMTCoordinates, cameraUMTCoordinates)) # (6.6)
    
    # projection on the image
    pointImageCoordinates = np.dot(K,pointCameraCoordinates) # (6.5)
    
    xImageCoordinates = pointImageCoordinates[0]/pointImageCoordinates[2]
    yImageCoordinates = pointImageCoordinates[1]/pointImageCoordinates[2]
    xPixelCoordinates = xImageCoordinates*imageSizeX/captorSizeX
    yPixelCoordinates = yImageCoordinates*imageSizeY/captorSizeY

    r = np.sqrt(((xImageCoordinates-px)/f)**2+((yImageCoordinates-py)/f)**2)

    xd = xImageCoordinates
    yd = yImageCoordinates
    
    # methode 1 de Wikipedia
    x_proj_wiki = xd + (xd-px)*(K1*r**2+K2*r**4+K3*r**6)+(T1*(r**2+2*(xd-px)**2)+2*T2*(xd-px)*(yd-py))
    y_proj_wiki = yd + (yd-py)*(K1*r**2+K2*r**4+K3*r**6)+(2*T1*(xd-px)*(yd-py)+T2*(r**2+2*(yd-py)**2))

    x_proj_wiki = x_proj_wiki*imageSizeX/captorSizeX
    y_proj_wiki = y_proj_wiki*imageSizeY/captorSizeY
    
    y_final = y_proj_wiki

    # If the pixel is within the image dimensions, it is kept. Otherwise, it returns -1
    x_final = 2*px_pixel - x_proj_wiki
    if not(0 <= (imageSizeX - xPixelCoordinates) <= imageSizeX and 0 <= yPixelCoordinates <= imageSizeY):
        x_final = -1
        y_final = -1
        
    return [x_final, y_final]

# **********************************************************************************************************************
        
def pylonsProjection(imPath, imName, write2file):   
    '''Returns the pixel coordinates of the top of all the pylons on an image
    '''
    maxDistanceToCamera = 500
    potentialPylons = openImages.getPotentialPylones(imPath, imName, maxDistanceToCamera, cameraParamFile)
    
    pylonsPixelCoordinates = []

    imageParameters = calculateImageParameters(imPath, imName)

    for singlePylon in potentialPylons:
        pylonsPixelCoordinates.append([singlePylon[0],projUMT2pixel(singlePylon[1:], imageParameters)])
    
    if write2file==True:
        pylonsId = [int(x[0]) for x in pylonsPixelCoordinates if x[1][0]>-1]
        pylonsCoordsX = [int(np.round(x[1][0])) for x in pylonsPixelCoordinates if x[1][0]>-1]
        pylonsCoordsY = [int(np.round(x[1][1])) for x in pylonsPixelCoordinates if x[1][0]>-1]
        
        
      
        with open(os.path.join('..', 'objectsOnImages', (imName[:-4] + '.csv')),'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)     
        
            for row in range(len(pylonsId)):
                csvRow = ['pyl', pylonsId[row], pylonsCoordsX[row], pylonsCoordsY[row]]
                spamwriter.writerow(csvRow)
       
    
    return pylonsPixelCoordinates

# **********************************************************************************************************************

def linesProjection(imPath, imName, write2file):
    '''Returns the pixel coordinates of all the lines on an image
    '''
    maxDistanceToCamera = 200
    potentialLinesPoints = openImages.getPotentialLinePoints(imPath, imName, maxDistanceToCamera, cameraParamFile)
    
    linesPixelCoordinates = []
    
    imageParameters = calculateImageParameters(imPath, imName)
    
    for singlePoint in potentialLinesPoints:
        linesPixelCoordinates.append(projUMT2pixel(singlePoint, imageParameters))
        
    if write2file==True:
        linesCoordsX = [int(np.round(x[0])) for x in linesPixelCoordinates if x[0]>-1]
        linesCoordsY = [int(np.round(x[1])) for x in linesPixelCoordinates if x[0]>-1]
        
        
          
        with open(os.path.join('..', 'objectsOnImages', (imName[:-4] + '.csv')),'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)     
        
            for row in range(len(linesCoordsX)):
                csvRow = ['lin', 0, linesCoordsX[row], linesCoordsY[row]]
                spamwriter.writerow(csvRow)
    
    return linesPixelCoordinates
    
# ______________________________________________________________________________________________________________________
# ======================================================================================================================
#                                               M A I N

def main():
    currentPath = os.path.dirname(os.path.abspath(__file__))
    path = currentPath.rsplit("/", 1)
    imPath = path[0] + '/data/gourd_c1818/Images/'

    imagesNames = os.listdir(imPath + '03/')
    imNumber = 1103

    projectedObjects = 'pylons'
    display = False
    
    plt.close('all')
    
    plt.figure()

    for imNumber in range(len(imagesNames)):
        if projectedObjects == 'lines' or 1==1:
            imName = imagesNames[imNumber]
            linesCoords = linesProjection(imPath, imName, True)
    
            if display :
                openImages.displayImageMPLT(imPath, imName)
                linesCoordsX = [x[0] for x in linesCoords if x[0]>-1]
                linesCoordsY = [x[1] for x in linesCoords if x[1]>-1]
                plt.plot(linesCoordsX, linesCoordsY, '.')
    
        if projectedObjects == 'pylons':
            imName = imagesNames[imNumber]
            pylonsCoords = pylonsProjection(imPath, imName, True)
    
            if display :
                openImages.displayImageMPLT(imPath, imName)
                pylonsCoordsX = [x[1][0] for x in pylonsCoords if x[1][0]>-1]
                pylonsCoordsY = [x[1][1] for x in pylonsCoords if x[1][0]>-1]
                plt.plot(pylonsCoordsX, pylonsCoordsY, 'or')

    plt.show()

# ______________________________________________________________________________________________________________________
# ======================================================================================================================

if __name__ == '__main__':
    main()