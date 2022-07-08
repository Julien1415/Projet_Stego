import numpy as np
import math
import numpy.random as npr
import matplotlib.pyplot as plt
import csv
import os
import time


from images_sauv import liste_images
from scipy.stats import gennorm
from etude_coefs import recup_image,selection_images
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta
from test_wave import decomposition_image, concatene
from fingerprinting import modif_image



os.chdir("C:/Users/fseignat/Desktop/stega Seignat/code_images")
start_time = time.time()

def simuler_GGD(N,sigma,beta):
    """
    

    Parameters
    ----------
    N : taille de l'échantillon
    sigma : paramètre de la GGD
    beta : paramètre de la GGD

    Returns
    -------
    renvoie une liste de taille correpondant aux valeurs de la loi GGD(sigma,beta)

    """
    r = gennorm.rvs(beta, scale=sigma,size=N)
    return r


def rapport_GGD(N,sigma,beta):
    """
    

    Parameters
    ----------
    N : taille des échantillons de la GGD
    sigma : paramètre de la GGD
    beta : paramètre de la GGD

    Returns
    -------
    retourne la valeur de la statistique

    """
    X=simuler_GGD(N, sigma, beta)
    Y=simuler_GGD(N, sigma, beta)
    num=np.sum(np.abs(X)**beta)
    den=np.sum(np.abs(Y)**beta)
    return num/den


def quantiles_experimentaux(M,N,sigma,beta,alpha):
    """
    

    Parameters
    ----------
    M : nombre de fois où l'on répète la procédure pour faire une moyenne
     N : taille des échantillons de la GGD
    sigma : paramètre de la GGD
    beta : paramètre de la GGD
    alpha : paramètre du quantile (entre 0 et 1)

    Returns
    -------
    renvoie la valeur moyenne du quantile de niveau alpha
    """
    L=[]
    for i in range(M):
        L.append(rapport_GGD(N, sigma, beta))
    quantile=np.quantile(L,alpha)
    return quantile



#Quantiles exacts


def quantiles_theoriques(N,beta,alpha):
    """

    Parameters
    ----------
    N : nombre de coefficients
    beta : paramètre GGD
    alpha : niveau de confiance du quantile (0.1,0.05,0.01)

    Returns
    -------
    renvoie le quantile théorique (nécessite une méthode de Newton)

    """
    b=2-4/(alpha**(beta/N))
    c=1
    Delta=b**2-4*c
    gamma=(-b+np.sqrt(Delta))/2
    return gamma 
    




##Quantiles TCL

def quantiles_TCL(N,beta,alpha):
    """
    Parameters
    ----------
    N : nombre de valeurs
    beta : paramètre GGD
    q : quantile de la N(0,1)  (alpha=0.1, q=1.282    alpha=0.05,q=1.645,    
                                alpha=0.01,q=2.326)

    Returns
    -------
    gamma : TYPE
        DESCRIPTION.

    """
    alpha_quantiles={0.1:1.282,0.05:1.645,0.01:2.326}
    q=alpha_quantiles[alpha]
    gamma=1/(1-beta*q**2/N)+np.sqrt(1/(beta*q**2/N-1)**2-1)
    return gamma



# L=np.arange(12000,500000,500)
# Q_th=[quantiles_theoriques(i,0.3, 0.05) for i in L]
# Q_TCL=[quantiles_TCL(i,0.3, 0.05) for i in L]
# plt.plot(L,Q_th,'r')
# plt.plot(L,Q_TCL,'b')
# plt.ylim([Q_th[0],Q_TCL[-1]])




Sigma=[1,2,15,50]
Beta=np.arange(0.5,1.6,0.1)
Alpha=[0.9,0.95,0.99]

# M=500
# N=50000

# with open('Quantiles.csv','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Sigma','Beta','Alpha','Quantile'])
#       writer.writerow('\n')
#       z=0
#       for sig in Sigma:
#          for bet in Beta:
#             bet=round(bet,1)
#             for alp in Alpha:
#                 z+=1
#                 print(z) 
#                 q=quantiles_experimentaux(M, N, sig, bet, alp)
#                 writer.writerow([sig,bet,alp,q])
#             writer.writerow('\n')



file=open('ALASKA/Quantiles_300000.csv','r')
r = csv.reader(file, delimiter=',')
liste = list(r)
file.close()


#S=selection_images(200,"Cover")
S=liste_images()
nb_echelle=3
c=0

# for i in S:
#     img=recup_image(i,"Cover")
#     taille=np.size(img)
#     #img_Stego=recup_image(i,"JUNIWARD")
#     Signe="Cover"
#     for e in range(nb_echelle):
#         WR=concatene(img,nb_echelle,e+1)
        
        
#         val_dep=val_dep_beta(WR)
#         beta_estim=beta_chap(WR,val_dep,10**(-4))
#         #sigma_estim=sigma_chap(WR,beta_estim)
#         if beta_estim!="Pas de solution" and beta_estim>=0.3 and beta_estim<=1.78:
                
#             den=np.sum(np.abs(WR)**beta_estim)

            
#             img_Stego=modif_image(img,0,0)
#             #img_Stego=modif_image(img_Stego,1,0.9)
#             #img_Stego=modif_image(img_Stego,2,0.9)
#             WR_Stego=concatene(img_Stego,nb_echelle,e+1)
#             num=np.sum(np.abs(WR_Stego)**beta_estim)
#             rapport=num/den
            
#             #Quantiles théoriques
#             #Q=quantiles_theoriques(taille,beta_estim,0.05)
            
            
#             #Quantiles estimés
#             #bet_round=float(round(0.02 * np.round(beta_estim/ 0.02),2))   #permet d'arrondir à 0.02 près (pour avoir les quantiles estimés)
#             #ind=int(round((round(bet_round-0.3,2)*100)*2))        #problème d'arrondi (aucune idée d'où ça vient)
#             #Q=float(liste[3+ind][2])
            
#             #Quantiles TCL
#             Q=quantiles_TCL(np.size(WR),beta_estim,0.05)
#             if rapport>Q:
#                 Signe="Stego"
#     if Signe=="Stego":
#         c+=1
#     print(Signe)

# print("\n")
# print(c/200)

##Calcul erreur



# Bet=[float(liste[2+i*4][0]) for i in range(75)]

# for k in range(len(Alpha)):
#     alpha=0.90#Alpha[k]
#     for j in range(len(Bet)):
#         erreur=0
#         beta=Bet[j]
#         Q=quantiles_theoriques(100000,beta,1-alpha)#float(liste[k+2+4*j][2])
#         for i in range(500):
#             rapport=quantiles_experimentaux(1,100000,1,beta,alpha)
#             if rapport>Q:
#                 erreur+=1
#         print("alpha= ",Alpha[k],"beta= ",round(beta,2),"pourcentage_erreur= ",erreur/500)
#     print("\n")
    


Moy=500
Nb=300000
sig=1

# BETA=np.arange(0.3,1.8,0.02)
# Alpha=[0.9,0.95,0.99]
# with open('Quantiles_100000.csv','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Beta','Alpha','Quantile'])
#       writer.writerow('\n')
#       z=0
#       for bet in BETA:
#           bet=round(bet,1)
#           for alp in Alpha:
#               z+=1
#               print(z)
#               q=quantiles_experimentaux(Moy, Nb,sig, bet, alp)
#               writer.writerow([bet,alp,q])
#           writer.writerow('\n')






# Quant_90=[float(liste[2+i*4][2]) for i in range(75)]
# Quant_95=[float(liste[3+i*4][2]) for i in range(75)]
# Quant_99=[float(liste[4+i*4][2]) for i in range(75)]


# plt.figure()
# plt.plot(Bet,Quant_90)
# plt.xlabel("Beta")
# plt.ylabel("Valeur du quantile")
# plt.title("Quantile à 90% en fonction de beta (sigma=1)")


# plt.figure()
# plt.plot(Bet,Quant_95)
# plt.xlabel("Beta")
# plt.ylabel("Valeur du quantile")
# plt.title("Quantile à 95% en fonction de beta (sigma=1)")


# plt.figure()
# plt.plot(Bet,Quant_99)
# plt.xlabel("Beta")
# plt.ylabel("Valeur du quantile")
# plt.title("Quantile à 99% en fonction de beta (sigma=1)")

