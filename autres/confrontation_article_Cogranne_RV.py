import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import random
import csv

from scipy.stats import norm
from confrontation_article_Cogranne_chi2 import k_barre,message,modif_LSB
from etude_coefs import recup_image,selection_images
import time
from fingerprinting import modif_image


os.chdir("C:/Users/fseignat/Desktop/stega Seignat/code_images/ALASKA")

start_time = time.time()
def po(x,mu,sigma):
    '''
    

    Parameters
    ----------
    x : valeur du pixel
    mu : moyenne du pixel (à estimer)
    sigma : écart-type du pixel (à estimer)

    Returns
    -------
    renvoie po(x) (avec les notations de l'article) correspondant à P(x_(m,n))=k

    '''
    return norm.pdf(x,mu,sigma)



def p1(x,mu,sigma,beta):
    """
    Parameters
    ----------
    x : valeur du pixel
    mu : moyenne du pixel (à estimer)
    sigma : écart-type du pixel (à estimer)
    beta : taux d'insertion (rapport entre le nombre de bits du message et le nombre total de pixels)
    
    Returns
    -------
    renvoie p1 (avec les notations de l'article)

    """
    return (1-beta/2)*norm.pdf(x,mu,sigma)+beta/2*norm.pdf(k_barre(x),mu,sigma)



def zone_pixel(img,taille,i,j):
    """
    

    Parameters
    ----------
    img : image à traiter
    taile : taille du carré à sélectionner autour de la valeur i,j

    Returns
    -------
    renvoie la zone à sélectionner
    """
    if i>taille/2 and i<np.shape(img)[0] and j>taille/2 and j<np.shape(img)[1]:
        return img[i-taille//2:i+taille//2+1,j-taille//2:j+taille//2+1]
    else:
        return img[i-np.minimum(0,np.abs(i-taille//2)):i+np.minimum(511,i+taille//2)+1,j-np.minimum(0,np.abs(j-taille//2)):j+np.minimum(511,j+taille//2)+1]    
    


def sigma_est(img):
    """
    

    Parameters
    ----------
    img : image à traiter (seul la partie bleue est touchée)

    Returns
    -------
    renvoie l'écart-type de l'image

    """

    return np.std(img)
    
    
def mu_est(img):
    """
    

    Parameters
    ----------
    img : image à traiter (seule la partie bleue est changée)

    Returns
    -------
    renvoie la moyenne de l'image

    """
    return np.mean(img)
    
    


# def test_RV(img,message):
#     """
    

#     Parameters
#     ----------
#     img : image test (on change seulement la partie bleue)
#     message : suite de bits à insérer

#     Returns
#     -------
#     valeur du RV

#     """
#     img=img[:,:,2]             #on ne change que sur la partie bleue 
#     L=len(message)
#     long=np.shape(img)[0]
#     lar=np.shape(img)[1]
#     beta=L/(long*lar)     #nombre de bits insérés par pixels
#     sigma=sigma_est(img)
#     mu=mu_est(img)
#     p=1
#     for m in range(long):
#         for n in range(lar):
#             val=img[m,n]
#             proba=po(val,mu,sigma)
#             proba_barre=po(k_barre(val),mu,sigma)
#             p*=(1-beta)+beta*(proba+proba_barre)/(2*proba)
#     return p 
    

def log_RV(img,message,zone=20):
    """
    

    Parameters
    ----------
    img : image test (on change seulement la partie bleue)
    message : suite de bits à insérer
    zone : taille du carré pour l'approximation de la moyenne et l'écart type

    Returns
    -------
    log du RV
    """
    I=img[:,:,2]             #on ne change que sur la partie bleue 
    L=len(message)
    long=np.shape(img)[0]
    lar=np.shape(img)[1]
    beta=L/(long*lar)     #nombre de bits insérés par pixels
    print("beta=",beta)
    s=0
    for m in range(long):
        for n in range(lar):
            img=zone_pixel(I,zone,m,n)
            sigma=sigma_est(img)
            mu=mu_est(img)
            val=I[m,n]
            if sigma!=0:
                t=beta*(val-mu)*(val-k_barre(val))/(2*sigma**2)
                s+=t
    return s



def seuil(alpha0):
    """
    

    Parameters
    ----------
    alpha0 : taux de faux positifs

    Returns
    -------
    renvoie le seuil tho tel que P(G>Tho)=alpha0 avec G une N(0,1)

    """
    erreur=10**(-4)
    tho_k=0
    tho_k_1=tho_k-(alpha0-1+norm.cdf(tho_k))/norm.pdf(tho_k)
    while np.abs(tho_k-tho_k_1)>erreur:
        tho_k=tho_k_1
        tho_k_1=tho_k-(alpha0-1+norm.cdf(tho_k))/norm.pdf(tho_k)
    return tho_k_1


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


S=selection_images(100,"Cover")
I=np.arange(0,1.2,0.2)
Inser=np.append(I,1.1)


# Tho=seuil(0.001)                    #0.1% de faux négatifs
# print("Tho=",Tho)
# L=[]
# for i in S:
#     img=recup_image(i,"Cover")
#     if type(img)!=bool:
#         #img_mod=modif_image(img,0,0.5)
#         #img_mod=modif_image(img_mod,1,0)
#         print("i=",i)
#         M=message(24000)
#         img=modif_LSB(img, M)
#         RV=log_RV(img,M)
#         L.append(RV)

# plt.hist(L,bins='auto',density=True)

##Fichiers déterminant si une image est Cover ou Stego




# with open('Tests_article_substitution.csv','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Image','Taux d\'insertion','Résultat'])
#       writer.writerow('\n')
#       z=0
#       Tho=seuil(0.001) 
#       for c in S:
#         print("c=",c)
#         img=recup_image(c,"Cover")
#         for taux in Inser:
#             z+=1
#             print(z)
#             if taux==1.1:                           #inversion de bits, on inverse tous les LSB
#                 img_mod=modif_image(img,0,1)
#                 mess=message(np.shape(img_mod)[0]*np.shape(img_mod)[1])
#                 res=test(log_RV(img_mod,mess)<Tho)
#                 writer.writerow([c,"inversion",res])
#             else:
#                 lg_mess=int(np.shape(img)[0]*np.shape(img)[1]*taux)
#                 mess=message(lg_mess)
#                 I=modif_LSB(img,mess)
#                 res=test(log_RV(I,mess)<Tho)
#                 writer.writerow([c,str(taux),res])
#         writer.writerow('\n')
 
    
# print("--- %s seconds ---" % (time.time() - start_time))


def sum_img_mu(img,zone=20):
    """
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    zone : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    permet d'éviter de recalculer plusiers fois la même quantité

    """
    I=img[:,:,2]
    long=np.shape(img)[0]
    lar=np.shape(img)[1]
    rho=0
    L=[]
    for m in range(long):
        for n in range(lar):
            img=zone_pixel(I,zone,m,n)
            sigma=sigma_est(img)
            L.append(sigma)
    L=np.array(L)        
    return(np.sum((1./L)**2))

def puissance(img,alpha0,beta):
    """
    

    Parameters
    ----------
    img : image à traiter
    alpha0 : taux de faux-positifs
    beta : taux d'insertion

    Returns
    -------
    puissance du test

    """
    zone=20
    I=img[:,:,2]
    tho=seuil(alpha0)
    long=np.shape(img)[0]
    lar=np.shape(img)[1]
    rho=sum_img_mu(img,zone)
    rho=np.sqrt(rho*beta**2/4)
    return 1-norm.cdf(tho-rho)



beta=0.09
alpha=[10**(-i) for i in range(20,-1,-1)]
#alpha=np.linspace(10**(-2),1,20)
img=recup_image(400,"Cover")
P=[puissance(img,i,beta) for i in alpha]  #Courbe 7.4.c (les puissances sont très différentes suivant les images)


# A=np.linspace(0,1,100)
# Seuil=[seuil(i) for i in A]
# plt.xlim([-5,5])
# plt.plot(Seuil,A)                         #Courbe 7.4.b



# Test=open('Tests_article_substitution.csv','r')
# T = csv.reader(Test, delimiter=',')
# liste_t = list(T)
# Test.close()

# L=[liste_t[j+i*8] for j in range(3,8) for i in range(0,500)]
# c=0
# for e in L:
#     if e[2]=='Cover':
#         c+=1
        
# print(c/len(L))