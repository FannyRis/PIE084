# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:45:39 2018

@author: Louis BAETENS
"""

import json
from pprint import pprint
import matplotlib.pyplot as plt
import csv
import numpy as np

              
def getDistancePortee(portee):
    '''depuis une portee, donne sa longueur depuis les coords pylones en
    X et Y'''
    vector = [portee["poleB"][1]-portee["poleA"][1], portee["poleB"][2]-portee["poleA"][2]]
    distance = np.linalg.norm(vector)
    return distance    

def getAnglePortee(portee):
   '''Retourne l'angle de la portee par rapport à l'axe Est, sens trigo'''
   vector = [portee["poleB"][1]-portee["poleA"][1], portee["poleB"][2]-portee["poleA"][2]]
     
   angle = np.arctan2(vector[1], vector[0]) #in radians
   print(angle*180/np.pi)
   return angle
    
def getGlobalPoints(fileName, step):
    '''Write the global coordinates UTM31 for all the lines, discretized every step
    '''
    path = "C:/Users/Louis/Google Drive/SUPAERO/PIE/gourd_c1818/reseau/"
    filePath = path+fileName

    data = json.load(open(filePath)) #load the data

    print(len(data)) #number of portees

#    step = 0.5 #longueur de la discretization, en metres
    
    # discretize chaque portee et calcule des points sur elle
    for itePortee in range(1):#range(len(data)):
        portee = data[itePortee]
#        pprint(portee)
        
        distancePortee = getDistancePortee(portee)
        anglePortee = getAnglePortee(portee)
        
        Xlocal = np.arange(0, distancePortee, step)
        X2local = np.power(Xlocal, 2)
        

        for subline in (["Right", "Center", "Left"]):
            Yparams = portee[subline]["Y"]
            Zparams = portee[subline]["Z"]
            
            Ylocal = Yparams["cy0"]+ Yparams["cy1"]*Xlocal+ Yparams["cy2"]*X2local
            Zlocal = Zparams["cz0"]+ Zparams["cz1"]*Xlocal+ Zparams["cz2"]*X2local
                            
            #transformer ces coords en coords globales UTM31
            Zglobal = Zlocal + portee["poleA"][3]  
            
            Xglobal = np.cos(anglePortee)*Xlocal - np.sin(anglePortee)*Ylocal + portee["poleA"][1]
            Yglobal = np.sin(anglePortee)*Xlocal + np.cos(anglePortee)*Ylocal + portee["poleA"][2]

            
            plt.plot(Xglobal,Zglobal, '.')
            with open(('linesGPS/' + str(itePortee) + '_' + subline + '.csv'), 'w') as f:            
                writer = csv.writer(f)
                for ptIte in range(len(Xglobal)):
                    writer.writerow([Xglobal[ptIte],Yglobal[ptIte],Zglobal[ptIte]])       
            
            
            
        
     
        
        
    
def main():
    getGlobalPoints("ranges_modelisation.json", 0.5)
#    writePylonsCoords("ranges_modelisation.json")
        
#    pprint(data[1])    
#    pprint(data[1])
    


if __name__ == '__main__':
    main()