import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time
import os
from random import random


def modif_image(img,clef,seuil):
    """
    Parameters
    ----------
    img : image à modifier
    clef : profondeur de bit (va de 0 à 7)
    seuil : représente le taux d'insertion. Si tirage>seuil, le bit est modifié

    Returns
    -------
    image modifiée
    """
    #Tatouage sur la partie bleue
    plan = np.zeros(img.shape[0:2], dtype="uint8")
    plan = img[:,:,2]

    #Tableau qui va déterminer les pixels à changer
    marque =  np.ones(np.shape(plan),dtype="uint8")*(2**clef)*(np.random.random(size=np.shape(plan))>seuil)

    #insertion de la marque
    plan_marque = np.bitwise_xor(plan,marque) #bitwise exclusive or
    img2 = np.copy(img)
    img2[:,:,2] = plan_marque

    return(img2)


def recup_image(num,BD):
    """
    Parameters
    ----------
    num : numéro de l'image à traiter
    BD : base de données (de type Cover ou JMiPOD)

    Returns
    -------
    renvoie l'image
    """
    L = ['0']*5
    S = list(str(num/10**4))
    S = np.delete(S,1)
    for i in range(len(S)):
        L[i] = S[i]
    s = ''.join(L)
    che = BD + "/" + str(s) + '.jpg'
    if os.path.exists(che):
        img = mpimg.imread(che)
        # img = np.asarray(Image.open(che))
    else:
        img = False     #le chemin n'existe pas (la base a des trous, 00006.jpg n'existe pas par exemple)
    return img