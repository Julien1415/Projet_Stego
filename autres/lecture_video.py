import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import time
import pandas as pd
import random
import pytube


from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from test_wave import decomposition_image, concatene
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta
from fingerprinting import modif_image


os.chdir("C:/Users/fseignat/Desktop/stega Seignat/Videos")
start_time = time.time()



# url="https://www.youtube.com/watch?v=mLr-9WJ3MpY"
# youtube = pytube.YouTube(url)
# video = youtube.streams.filter(res='1080p')
# video.first().download()





##Découpe vidéo

def decoupe(file_ori,file_fin,start_time,end_time):
    """
    

    Parameters
    ----------
    file_ori : adresse du film à couper
    file_fin : nom du film après découpe
    start_time : temps de début de découpe
    end_time : temps de fin de découpe

    Returns
    -------
    renvoie la vidéo découpée
    """
    ffmpeg_extract_subclip(file_ori,start_time,end_time, targetname=file_fin)


#decoupe("💣 TRUE ROMANCE Bande Annonce VF (1993).mp4","BA_TR.mp4",13,43)

## Lecture vidéo 



def tableau():   
    L=[]
    cap = cv.VideoCapture("BA_TR.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.IMREAD_COLOR)
        L.append(gray)                  #Liste qui contient tous les pixels qui différentes frames
        #cv.imshow('frame',gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    return L

#L=tableau()
#np.save('BA_TR.npy', L)



def cv_plt(T):
    """
    

    Parameters
    ----------
    T : tableau de pixels

    Returns
    -------
    permet la conversion d'un tableau qui s'affiche bien avec cv.imshow en la même
    image avec plt.imshow (marche aussi en sens inverse)

    """
    b,g,r = cv.split(T)  
    img2 = cv.merge([r,g,b])
    return img2



def distance(F1,F2):
    """
    

    Parameters
    ----------
    F1 : frame 1
    F2 : frame 2

    Returns
    -------
    renvoie la norme L² de la différence entre les 2 frames successives

    """
    F1=F1.astype("float64")             #conversion en float sinon uint_8 qui calcule modulo 256
    F2=F2.astype("float64")
    D=np.sum(np.abs(F1-F2)**2)
    return D
    



def recup_Cover(L,nb):
    """

    Parameters
    ----------
    L : liste des différentes frames
    nb : indice de l'image à traiter

    Returns
    -------
    renvoie l'image qui peut être  considéré comme Cover (soit nb-1, soit nb+1)
    en prenant le min de la distance (norme L² de la différence)

    """
    F0=L[nb]            #image à tester
    F1=L[nb-1]
    F2=L[nb+1]
    D1=distance(F0,F1)
    D2=distance(F0, F2)
    if D1<D2:
        return nb-1
    return nb+1



def modif_video(F,nb_modif):
    """
    

    Parameters
    ----------
    F : fichier avec les décompositions des différentes trames
    nb_modif : nombre d'images modifiées (à tirer aléatoirement)
    
    Returns
    -------
    renvoie un fichier F' du même type que F avec les trames modifiées

    """
    F_mod=np.copy(F)
    
    #Tirage aléatoire
    L=random.sample(range(len(F)), nb_modif)
    for ind in L:
        img=F[ind]
        
        #Modif de l'image (peut changer selon la profondeur et/ou seuil voulu)
        img_mod=modif_image(img,0,0)
        F_mod[ind]=img_mod
    return (F_mod,L)



def supp_val_extrem(WR,quantiles):
    """


    Parameters
    ----------
    WR : tableau des coefficients d'ondellettes
    quantiles : seuil (5%) pour éliminer les valeurs extrêmes (à gauche et à droite)

    Returns
    -------
    renvoie le liste sans les valeurs extrêmes

    """
    L=WR#np.reshape(WR,np.shape(WR)[0]*np.shape(WR)[1])
    P=WR[np.where((L>np.quantile(L,quantiles)) & (L<np.quantile(L,1-quantiles)))]
    return P




    