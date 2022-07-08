import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
import os
os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")    #chemin vers le dossier de travail

from wavelet2D import *


def decomposition_image(img,nb_echelles,echelle,plan):
    """
    Parameters
    ----------
    img : image à décomposer(ligne de type img=mpimg.imread('C:/Users/fseignat/Desktop/stega Seignat/code_images/images/2.jpg'))

    nb_echelles : nombre d'échelles de décomposition
    
    echelle : échelle que l'on veut étudier (va de 1 à nb_échelles)
    
    plan : plan à étudier (de 1 à 3, le 0 correspond aux 4 images de l'échelle)
    
    Returns
    -------
    WR : renvoie les coefficients d'ondelettes
    """
    
    f=makeonfilter('Daubechies',4)
     
    R=np.array(img[:,:,0],'float')
    G=np.array(img[:,:,1],'float')
    B=np.array(img[:,:,2],'float')
    
    #Décompsoition Gris
    Gris = R+G+B
    #Gris = np.random.randn(Gris.shape[0],Gris.shape[1])
    #plt.imshow(Gris)
    
    wctest = banc_filtre2d(0,Gris,nb_echelles,f)
    #plt.imshow(wctest)
    
    
    WR=Lire_Echelle_Mallat2D(wctest,echelle,plan)
    #plt.imshow(WR)
    
    
    #WR=np.zeros(WR.shape)
    
    #Reconstruction
    #Ecrire_Echelle_Mallat2D(wctest,WR,1,1)
    #imgtest = banc_filtre2d(1,wctest,3,f)
    #print(np.max(Gris-imgtest))
    #plt.imshow(np.abs(Gris-imgtest))
    
    #Décomposition par plan couleur
    #wcR = banc_filtre2d(0,R,3,f)
    #wcG = banc_filtre2d(0,G,3,f)
    #wcB = banc_filtre2d(0,B,3,f)
    #plt.imshow(wcG)
    
    
    #WRB=Lire_Echelle_Mallat2D(wcB,3,0)
    #plt.imshow(WRB)

    return WR


def concatene(img,nb_echelle,echelle):
    """

    Parameters
    ----------
    img :image à traiter
    nb_echelle : nombre d'échelles de décomposition
    échelle : échelle à traiter

    Returns
    -------
    renvoie la liste concaténée des coefficients d'ondelettes

    """
    WR1=decomposition_image(img,nb_echelle,echelle,1)
    WR2=decomposition_image(img,nb_echelle,echelle,2)
    WR3=decomposition_image(img,nb_echelle,echelle,3)
    return np.concatenate((WR1,WR2,WR3))