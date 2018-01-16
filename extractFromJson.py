# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:45:39 2018

@author: Louis BAETENS
"""

import json
from pprint import pprint
import matplotlib.pyplot as plt
import csv
import os

def writePylonsCoords(fileName):
    '''Write the coordinates of the pylons from the JSON file
    '''
    path = "C:/Users/Louis/Google Drive/SUPAERO/PIE/gourd_c1818/reseau/"
    filePath = path+fileName

    data = json.load(open(filePath)) #load the data
  
    # separate identical pylons
    pylonsId = []
    pylonsX = []
    pylonsY = []
    pylonsZ = []
    for pylon in range(len(data)):
        for pole in (["poleA", "poleB"]):
            if data[pylon][pole][0] not in pylonsId :
                pylonsId.append(data[pylon][pole][0])
                pylonsX.append(data[pylon][pole][1])
                pylonsY.append(data[pylon][pole][2])
                pylonsZ.append(data[pylon][pole][3])
                
    plt.plot(pylonsX, pylonsY, '.')
    plt.show()

    # ecrit le tout dans une ficheir csv
    with open('pylonsGPS.csv', 'w') as f:
        writer = csv.writer(f)
 
#        writer.writerows([pylonsId, pylonsX])
        
        for pylId in range(len(pylonsId)):
            writer.writerow([pylonsId[pylId], pylonsX[pylId], pylonsY[pylId], pylonsZ[pylId]])


               
def concatenateAllCSV():
    currentPath = os.path.dirname(os.path.abspath(__file__))
    linesPath = os.path.join(currentPath, "linesGPS")
    print(linesPath)
    
    linesGPSNames = os.listdir(linesPath)
    
    fout=open("all_linesGPS.csv","a")
    for lineName in linesGPSNames:
        for line in open(os.path.join(linesPath, lineName)):
             fout.write(line)   
             if line == None:
                 print(line)
    fout.close()



def showSomeExample():
    path = "C:/Users/Louis/Google Drive/SUPAERO/PIE/gourd_c1818/reseau/"
    filePath = path+"ranges_modelisation.json"

    data = json.load(open(filePath)) #load the data

    print(len(data)) #number of portees
    pylonsId = []
    for pylon in range(len(data)):
        pylA = int(data[pylon]["poleA"][0])
        pylB = int(data[pylon]["poleB"][0])
        pylonsId.append(pylA)
        pylonsId.append(pylB)

    pylonsId.sort() #range en ordre croissant
    print("Sorted : ")
    pprint(pylonsId)
    print("Number of distinct pylons / total number of pylons : ")
    print(str(len(set(pylonsId))) + "/" + str(len(pylonsId)))
    
def main():
#    showSomeExample()
#    writePylonsCoords("ranges_modelisation.json")
    concatenateAllCSV()
#    pprint(data[1])    
#    pprint(data[1])
    


if __name__ == '__main__':
    main()