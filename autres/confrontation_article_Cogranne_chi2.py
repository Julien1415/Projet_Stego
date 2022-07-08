import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import random
import collections
import csv

from math import ceil
from etude_coefs import recup_image
from scipy.stats import chi2,norm
from fingerprinting import modif_image
from test_wave import decomposition_image, concatene
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta

os.chdir("C:/Users/fseignat/Desktop/stega Seignat/code_images/ALASKA")

def compte_pixel(img):
    """
    

    Parameters
    ----------
    img : image à traiter

    Returns
    -------
    renvoie une liste contenant à l'indice i le nombre de pixels valant i dans l'image

    """
    occ=collections.Counter(img)            #renvoie le nombre d'occurences de l'indice i (via Occ[i])
    return occ


def k_barre(x):
    """
    

    Parameters
    ----------
    x : valeur du pixel
    
    Returns
    -------
    renvoie x+(-1)**x
    """
    return x+(-1)**x


def chi_deux(img):
    """
    

    Parameters
    ----------
    img : image à traiter

    Returns
    -------
    calcul la valeur du chi_deux (somme (Nk-Nk*)**2/Nk*)
    Je garde les notations de l'article
    """
    img=img[:,:,2]
    img=np.reshape(img,len(img)**2)     #convertir en une liste
    Occ=compte_pixel(img)
    som=0
    for pix in range(256):            #valeur des pixels entre 0 et 255
        pix_barre=k_barre(pix)
        Nk=Occ[pix]
        Nk_etoile=(Occ[pix]+Occ[pix_barre])/2
        if Nk_etoile!=0:
            som+=(Nk-Nk_etoile)**2/Nk_etoile
    return som


def seuil(p0,df):
    """
    Parameters
    ----------
    p0 : faux négatif (probabilité qu'une image stégo soit jugé Cover)
    df : degrés de liberté (égal à 127)
                       
    Returns
    -------
    renvoie le seuil T vérifiant P(khi_deux>T)=p0=1-P(khi_deux<T).
    On cherche donc le zéro de p0-(1-P(khi_deux<T))
    """
    erreur=10**(-4)
    tho_k=175
    tho_k_1=tho_k-(p0-1+chi2.cdf(tho_k,df))/chi2.pdf(tho_k,df)
    while np.abs(tho_k-tho_k_1)>erreur:
        tho_k=tho_k_1
        tho_k_1=tho_k-(p0-1+chi2.cdf(tho_k,df))/chi2.pdf(tho_k,df)
    return tho_k_1
    



def message(N):
    """
    

    Parameters
    ----------
    N : nombre de bits du message

    Returns
    -------
    crée un vecteur de taille contenant des 0 et des 1 distribuées suivant une loi uniforme

    """
    M=np.random.randint(2,size=N)
    return M


def aleatoire(N):
    """
    

    Parameters
    ----------
    N : nombre de valeurs à modifier

    Returns
    -------
    renvoie N valeurs différentes
    """
    c=0
    A=[]
    rac=ceil(np.sqrt(N))
    Abs=np.arange(512)
    Ord=np.arange(512)
    A=np.transpose([np.tile(Abs, len(Ord)), np.repeat(Ord, len(Abs))]) #tous les pixels possibles
    idx=random.sample(range(np.shape(A)[0]),N)
    Alea=A[idx,:]               #permet d'avoir des couples uniques de pixels à modifier
    return Alea

def insertion_bit(img,mess):
    """
    

    Parameters
    ----------
    img : image à modifier
    mes=message à intéger

    Returns
    -------
    renvoie un tableau avec les pixels à modifier (tableau de 0 et 1). 
    Ne sert que si le taux d'insertion est inférieur à 1
    """
    I=np.zeros((np.shape(img)[0],np.shape(img)[1]))
    nb_bit=0
    lg_mess=len(mess)
    A=aleatoire(lg_mess)
    while nb_bit<lg_mess:
        I[A[nb_bit]]=mess[nb_bit]
        nb_bit+=1
    return I
    
    

def modif_LSB(img,M):
    """
    

    Parameters
    ----------
    img : image à modifier
    M : suite de bits à insérer dans l'image

    Returns
    -------
    image modifiée par substitution de bits

    """
    I=img[:,:,2]
    T=insertion_bit(img,M)
    I=I-I%2+T
    img2=np.copy(img)
    img2[:,:,2]=I
    return(img2)

def test(B):
    """
    

    Parameters
    ----------
    B : booléen

    Returns
    -------
    si True renvoie Cover, sinon renvoie Stego

    """
    if B==True:
        return ("Cover")
    else:
        return ("Stego")




# Tho=seuil(0,127)                    #0.1% de faux négatifs
# print("Tho=",Tho)
# for i in range(0,200):
#     img=recup_image(i,"UERD")
#     if type(img)!=bool:
#         #img_mod=modif_image(img,0,0.5)
#         #img_mod=modif_image(img_mod,1,0)
#         #M=message(512**2)
#         #img=modif_LSB(img, M)
#         print(test(chi_deux(img)>Tho))


#Paramètres avec méthode de substitution
nb_echelle=3
S=[4913, 64164, 65768, 22590, 2627, 8563, 15716, 2139, 34515, 16586]
Taux=np.arange(0.1,1.1,0.1)

with open('Param_Substi.csv','a',newline='') as fichiercsv:
      writer=csv.writer(fichiercsv)
      writer.writerow(['Image','Echelle','Taux d\'insertion', 'Beta_Chapeau','Sigma_chapeau'])
      writer.writerow('\n')
      z=0
      for c in S:
        print("c=",c)
        img=recup_image(c,"Cover")
        for i in range(0,nb_echelle):
            for t in Taux:
                z+=1
                print(z)
                t=round(t,1) 
                for l in range(100):
                          B=[]
                          S=[]
                          #img_mod=modif_image(img,0,s)
                          #img_mod=modif_image(img_mod,1,s)
                          #img_mod=modif_image(img_mod,2,s)
                          M=message(round(512**2*t))
                          img_mod=modif_LSB(img, M)
                          WR=concatene(img_mod,nb_echelle,i+1)
                          val_dep=val_dep_beta(WR)
                      
                          beta_estim=beta_chap(WR,val_dep,10**(-4))
                          B.append(beta_estim)
                          sigma_estim=sigma_chap(WR,beta_estim)
                          S.append(sigma_estim)
                bet=np.mean(B)
                sig=np.mean(S)
                writer.writerow([str(c),str(i+1),str(t),str(bet),str(sig)])
        writer.writerow('\n')
