import openImages
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import os

class PhotoParameters:

    def __init__(self, cameraParamFile, imPath, imName):
        self.imageName = imageName
        self.cameraWorldCoordinates = cameraWorldCoordinates
        self.cameraRotation = cameraRotation

    def getPhotoParameters(self, cameraParamFile, imPath, imName):
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