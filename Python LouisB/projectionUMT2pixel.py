# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:55:39 2018

@author: Louis BAETENS
"""
import openImages
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import os

#def extractCoords():
    
def m2mm(measure):
    '''Convert a measure in meters into milimeters
    '''
    return measure/1000
    
def projUMT2pixel(imPath, imName, UMTcoordinates):
    '''Project a UMT31 point on the image, given its parameters
    and those of the camera. Returns the pixel coordinates of the projected point
    '''
    
    (imageName,Xcam,Ycam,Zcam,omega,phi,kappa) = openImages.getPhotoParameters(imPath, imName)
    (captorSizeX, captorSizeY, focal, px, py, K1, K2, K3, T1, T2) = openImages.getCameraParameters(imPath)
    
#    print(imageName,Xcam,Ycam,Zcam,omega,phi,kappa)
#    print('\n \n')
#    print (captorSizeX, captorSizeY, focal, px, py, K1, K2, K3, T1, T2)
    
#    img = mpimg.imread(os.path.join(imPath,'03',imName))
#    (m,n,d) = img.shape
#    print(len(img))
    m = 2048
    n = 2448
    
   
    captorSizeX = m2mm(captorSizeX) #convert into mm
    captorSizeY = m2mm(captorSizeY)
    f = m2mm(focal)
    px = m2mm(px)
    py = m2mm(py)
    
    
    C = [Xcam, Ycam, Zcam]
    
    omega = np.deg2rad(omega)
    phi = np.deg2rad(phi)
    kappa = np.deg2rad(kappa)
    
    R_kappa = [[cos(kappa), sin(kappa), 0], [-sin(kappa), cos(kappa), 0], [0, 0, 1]]
    R_omega = [[1,0,0], [0, cos(omega), sin(omega)], [0, -sin(omega), cos(omega)]]
    R_phi = [[cos(phi), 0, -sin(phi),], [0, 1, 0], [sin(phi), 0, cos(phi)]]
    
    R = np.dot(np.dot(R_kappa,R_phi),R_omega)
    
#    print(R)
    
    px_pixel = px*n/captorSizeX;
    py_pixel = py*m/captorSizeY;
    K = [[f,0,px], [0, f, py],[0, 0, 1]]
    
    #cgt de repere UTM31 -> camera
    
    Xcam =  np.dot(R, np.subtract(UMTcoordinates, C)) # (6.6)
    
    #projection sur image
    x = np.dot(K,Xcam) # (6.5)
    
    x_proj = x[0]/x[2]; # coord du point sur l'image
    y_proj = x[1]/x[2];
    x_proj_pixel = x_proj*n/captorSizeX;
    y_proj_pixel = y_proj*m/captorSizeY;
    
    
    r = np.sqrt(((x_proj-px)/f)**2+((y_proj-py)/f)**2)


    xd = x_proj
    yd = y_proj
    
    # methode 1 de Wikipedia
    x_proj_wiki = xd + (xd-px)*(K1*r**2+K2*r**4+K3*r**6)+(T1*(r**2+2*(xd-px)**2)+2*T2*(xd-px)*(yd-py))
    y_proj_wiki = yd + (yd-py)*(K1*r**2+K2*r**4+K3*r**6)+(2*T1*(xd-px)*(yd-py)+T2*(r**2+2*(yd-py)**2))

    x_proj_wiki = x_proj_wiki*n/captorSizeX
    y_proj_wiki = y_proj_wiki*m/captorSizeY
    
    x_final = 2*px_pixel - x_proj_wiki
    y_final = y_proj_wiki
    
#    x_final = n-x_proj_pixel
#    y_final = y_proj_pixel
    

    if((n-x_proj_pixel)>=0 and (n-x_proj_pixel)<=n and y_proj_pixel>=0 and y_proj_pixel<=m):
        #if the pixel is within the image dimensions, it is kept. Otherwise,
        #it returns -1
#    if(x_final>=0 and x_final<=n and y_final>=0 and y_final<=m):

        print(x_final, y_final)
        
    else:
        x_final = -1
        y_final = -1
        
    return [x_final, y_final]




        
def pylonsProjection(imPath, imName):   
    maxDistanceToCamera = 500
    potentialPylons = openImages.getPotentialPylones(imPath, imName, maxDistanceToCamera)
    
    pylonsCoordinates = [x[1:] for x in potentialPylons]
    pylonsPixelCoordinates = []
#    print(pylonsCoordinates)
    
    for singlePylon in potentialPylons:
        print(singlePylon[0])
        pylonsPixelCoordinates.append([singlePylon[0],projUMT2pixel(imPath, imName, singlePylon[1:])])
    
    return pylonsPixelCoordinates#pylonsCoordinates
    
def linesProjection(imPath, imName):   
    maxDistanceToCamera = 500
    potentialLinesPoints = openImages.getPotentialLinePoints(imPath, imName, maxDistanceToCamera)
    
#    pylonsCoordinates = [x[1:] for x in potentialPylons]
    linesPixelCoordinates = []
#    print(pylonsCoordinates)
    
    for singlePoint in potentialLinesPoints:
#        print(singlePylon[0])
        linesPixelCoordinates.append(projUMT2pixel(imPath, imName, singlePoint))
    
    return linesPixelCoordinates#pylonsCoordinates
    

def main():
    plt.close('all')
    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, "gourd_c1818\\Images\\")
    
    imPath = "C:\\Users\\Louis\\Google Drive\\PIE Supaero (1)\\PIE_supaero_2017\\data\\gourd_c1818\\Images\\"
    print(imPath)
    
    imagesNames = os.listdir(os.path.join(imPath, '03'))[1:-2]
    imagesNumber = np.random.randint(0, len(imagesNames), size=2)

    
#    print(imagesNames)
    for imNum in imagesNumber:
#        imName = "03_00000" + str(imNum) + ".jpg"
        imName = imagesNames[imNum]
        print(imName)
#        imName = "03_00000748.jpg"
#        pylonsCoords = pylonsProjection(imPath, imName)
#        
#        print(pylonsCoords)
#        
#        wannaPlot = True
#        
#        if wannaPlot == True:
#            plt.figure()
#
#            openImages.displayImageMPLT(imPath, imName)
#            pylonsId = [x[0] for x in pylonsCoords if x[1][0]>-1]
#            pylonsCoordsX = [x[1][0] for x in pylonsCoords if x[1][0]>-1]
#            pylonsCoordsY = [x[1][1] for x in pylonsCoords if x[1][0]>-1]
#            plt.scatter(pylonsCoordsX, pylonsCoordsY)
            
            
        linesCoords = linesProjection(imPath, imName)
        
        print(linesCoords)
        
        wannaPlot = True
        
        if wannaPlot == True:
            plt.figure()

            openImages.displayImageMPLT(imPath, imName)
#            pylonsId = [x[0] for x in pylonsCoords if x[1][0]>-1]
            linesCoordsX = [x[0] for x in linesCoords if x[0]>-1]
            linesCoordsY = [x[1] for x in linesCoords if x[1]>-1]
            plt.scatter(linesCoordsX, linesCoordsY)

        
        
    
#    UMTcoord = [372904.4705323878,  4961525.41389998,   176.55100000000002]
    
    
#    projUMT2pixel(imPath, imName, UMTcoord)


if __name__ == '__main__':
    main()
#    cProfile.run('main()', 'tgz.txt')
#    import pstats
#    p = pstats.Stats('tgz.txt')
#    p.strip_dirs().sort_stats('cumtime').print_stats(20)