import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gennorm
import matplotlib.image as mpimg
import time
import datetime
import os
import pandas as pd
from sklearn.cluster import KMeans
os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")    #chemin vers le dossier de travail

import wavelet2D
from fingerprinting import *
from test_wave import concatene, decomposition_image
from estimation_parametres import *

## Traitement sur une image (calcul des coefs d'ondelettes, estimation, calcul de la pval, affichage de l'histogramme/densité, probplot)
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
num = 8422
num = 3
img = recup_image(num,BD)
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/lena_gray.jpeg')

echelle = 1
WR = concatene(img,1,echelle)
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
sigma_estim=sigma_chap(WR,beta_estim)
T=np.reshape(WR,WR.shape[0]*WR.shape[1])
T_tri = np.sort(T)
taille_T = len(T_tri)

pval = stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim))[1]
# stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim))

plt.figure()
plt.hist(T, bins='auto',density=True, color='red', alpha=0.5)
x = np.linspace(np.min(WR),np.max(WR),500)
plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'b-', lw=1.5, alpha=0.3)
plt.title(f"Cover {num}, beta = {round(beta_estim,5)}, sigma = {round(sigma_estim,5)}")
plt.xlabel(f"pvalue = {pval}")
# plt.show()
# plt.close('all')

# Problot
ech_ggd = np.sort(gennorm.rvs(beta_estim, scale=sigma_estim, size=taille_T))
# stats.probplot(ech_ggd,dist='gennorm',sparams=(beta_estim,0,sigma_estim),plot=plt)
plt.figure()
plt.scatter(T_tri,ech_ggd,c='k',s=6,alpha=0.5)
plt.plot(T_tri,T_tri,'r')
plt.plot(ech_ggd,ech_ggd,'r')
plt.title(f"Coeffs contre échantillon GGD pour l'image {num}, pval={round(pval,10)}")
plt.xlabel(f"Coeffs d'ondelettes,  pval={pval}")
plt.ylabel("Echantillon GGD")
plt.show()

## Impact sur la pvalue d'un filtrage des valeurs extremes des coeffs
num = 8422
num = 3     #la 4 est intéressente
# Bd = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/lena_gray.jpeg'
# img_1 = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/lena_gray.jpeg')
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
img = recup_image(num,BD)

echelle = 1
WR = concatene(img,2,echelle)
T=np.reshape(WR,WR.shape[0]*WR.shape[1])
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
sigma_estim=sigma_chap(WR,beta_estim)

T_tri = np.sort(T)
taille_T = len(T_tri)
pourcent = 5
perc = int(taille_T*(pourcent/100)/2)
T_filtre = T_tri[perc:-perc] #filtre {pourcent}% des valeurs
taille_T_filtre = len(T_filtre)
pval = stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim))[1]
pval_filtre = stats.kstest(T_filtre, 'gennorm', (beta_estim,0,sigma_estim))[1]  #test de T_flitre contre les estimations de T

# Pas de filtre
plt.figure()
plt.hist(T, bins='auto',density=True, color='red', alpha=0.5)
x = np.linspace(np.min(T),np.max(T),500)
plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'b-', lw=1.5, alpha=0.3)
plt.title(f"Cover {num}, beta = {round(beta_estim,5)}, sigma = {round(sigma_estim,5)}")
plt.xlabel(f"pvalue = {pval}")
# plt.show()
# plt.close('all')

# Filtre sur T sans recalcul de pval
plt.figure()
plt.hist(T_filtre, bins='auto',density=True, color='red', alpha=0.5)
x = np.linspace(np.min(T_filtre),np.max(T_filtre),500)
plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'b-', lw=1.5, alpha=0.3)
plt.title(f"Image {num} filtrée à {pourcent}%, beta = {round(beta_estim,6)}, sigma = {round(sigma_estim,6)}")
plt.xlabel(f"pvalue = {pval_filtre}")
# plt.show()
# plt.close('all')

# Recalcul de la pvalue par rapport à T_filtre
val_dep_filtre=val_dep_beta(T_filtre)
beta_estim_filtre=beta_chap(T_filtre,val_dep_filtre,10**(-4))
sigma_estim_filtre=sigma_chap(T_filtre,beta_estim_filtre)
pval_filtre2 = stats.kstest(T_filtre, 'gennorm', (beta_estim_filtre,0,sigma_estim_filtre))[1]  #test de T_flitre contre les estimations de T_filtre
plt.figure()
plt.hist(T_filtre, bins='auto',density=True, color='red', alpha=0.5)
x = np.linspace(np.min(T_filtre),np.max(T_filtre),500)
plt.plot(x, gennorm.pdf(x,beta_estim_filtre,scale=sigma_estim_filtre),'b-', lw=1.5, alpha=0.3)
plt.title(f"Image {num} filtrée à {pourcent}%, beta_filtre = {round(beta_estim_filtre,6)}, sigma_filtre = {round(sigma_estim_filtre,6)}")
plt.xlabel(f"pvalue = {pval_filtre2}")
plt.show()
# plt.close('all')

## Problot après filtrage
ech_ggd_filtre = gennorm.rvs(beta_estim, scale=sigma_estim, size=taille_T_filtre)
ech_ggd_filtre = np.sort(ech_ggd_filtre)
# plt.figure()
# stats.probplot(ech_ggd_filtre,dist='gennorm',sparams=(beta_estim,0,sigma_estim),plot=plt)
plt.figure()
plt.scatter(T_filtre,ech_ggd_filtre,c='k',s=6,alpha=0.5)
# plt.plot(T_filtre,T_filtre,'b')
plt.plot(ech_ggd_filtre,ech_ggd_filtre,'r')
# plt.plot(np.linspace(min(ech_ggd_filtre),max(ech_ggd_filtre)),np.linspace(min(ech_ggd_filtre),max(ech_ggd_filtre)))
plt.title(f"Coeffs contre échantillon GGD pour l'image {num}")
plt.xlabel(f"Coeffs d'ondelettes,  pval={pval_filtre}")
plt.ylabel("Echantillon GGD")
plt.show()

## Automatisation avec filtre
idx_filtre = ["beta1", "sigma1", "pvalue1", "beta2", "sigma2", "pvalue2", "beta1_filtre", "sigma1_filtre", "pvalue1_filtre", "beta2_filtre", "sigma2_filtre", "pvalue2_filtre"]    # liste des indices pour le dataframe pandas


def param_filtre(num ,path = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'):
    """
    Parameters
    ----------
    num : numéro de l'image à traiter
    path : chemin vers dossier contenant la banque

    Returns
    -------
    renvoie différents parametres de l'image (notammment les sigma/beta
    des 2 premieres échelles, les sigma/beta de l'image modifiée en
    fonction du seuil et de la profondeur d'insertion...) associée au
    numéro, si elle existe
    """

    img = recup_image(num,path)
    L = []
    pourcent = 5

    if type(img) == bool :
        return [np.nan]*len(idx_filtre)

    else :
        # paramètres sigma/beta et pvalue à l'échelle 1
        WR_1 = concatene(img,1,1)
        val_dep_1 = val_dep_beta(WR_1)
        beta_estim_1 = beta_chap(WR_1,val_dep_1,10**(-4))
        sigma_estim_1 = sigma_chap(WR_1,beta_estim_1)
        # print(sigma_estim_1)

        T_1 = np.reshape(WR_1,WR_1.shape[0]*WR_1.shape[1])
        if sigma_estim_1 == 0 or beta_estim_1 == np.nan :
            pval_1 = np.nan
        else :
            pval_1 = stats.kstest(T_1, 'gennorm', (beta_estim_1,0,sigma_estim_1))[1]
        L = L + [round(beta_estim_1,7), round(sigma_estim_1,7), pval_1]

        # paramètres sigma/beta et pvalue à l'échelle 2
        WR_2 = concatene(img,2,2)
        val_dep_2 = val_dep_beta(WR_2)
        beta_estim_2 = beta_chap(WR_2,val_dep_2,10**(-4))
        sigma_estim_2 = sigma_chap(WR_2,beta_estim_2)
        # print(sigma_estim_2)

        T_2 = np.reshape(WR_2,WR_2.shape[0]*WR_2.shape[1])
        if sigma_estim_2 == 0 or beta_estim_2 == np.nan :
            pval_2 = np.nan
        else :
            pval_2 = stats.kstest(T_2, 'gennorm', (beta_estim_2,0,sigma_estim_2))[1]
        L = L + [round(beta_estim_2,7), round(sigma_estim_2,7), pval_2]


        # paramètres sigma/beta et pvalue à l'échelle 1, filtré
        WR_1 = concatene(img,1,1)
        T_1 = np.sort(np.reshape(WR_1,WR_1.shape[0]*WR_1.shape[1]))
        perc = int(len(T_1)*(pourcent/100)/2)
        T_filtre_1 = T_1[perc:-perc] #filtre {pourcent}% des valeur
        val_dep_1 = val_dep_beta(T_filtre_1)
        beta_estim_1 = beta_chap(T_filtre_1,val_dep_1,10**(-4))
        sigma_estim_1 = sigma_chap(T_filtre_1,beta_estim_1)

        if sigma_estim_1 == 0 or beta_estim_1 == np.nan :
            pval_1 = np.nan
        else :
            pval_1 = stats.kstest(T_filtre_1, 'gennorm', (beta_estim_1,0,sigma_estim_1))[1]
        L = L + [round(beta_estim_1,7), round(sigma_estim_1,7), pval_1]

        # paramètres sigma/beta et pvalue à l'échelle 2, filtré
        WR_2 = concatene(img,2,2)
        T_2 = np.sort(np.reshape(WR_2,WR_2.shape[0]*WR_2.shape[1]))
        perc = int(len(T_2)*(pourcent/100)/2)
        T_filtre_2 = T_2[perc:-perc] #filtre {pourcent}% des valeur
        val_dep_2 = val_dep_beta(T_filtre_2)
        beta_estim_2 = beta_chap(T_filtre_2,val_dep_2,10**(-4))
        sigma_estim_2 = sigma_chap(T_filtre_2,beta_estim_2)

        if sigma_estim_2 == 0 or beta_estim_2 == np.nan :
            pval_2 = np.nan
        else :
            pval_2 = stats.kstest(T_filtre_2, 'gennorm', (beta_estim_2,0,sigma_estim_2))[1]
        L = L + [round(beta_estim_2,7), round(sigma_estim_2,7), pval_2]

        return L


def array_param_filtre(S):
    """
    Parameters
    ----------
    S = liste des numéros des images à traiter

    Returns
    -------
    La matrice dont la n-ième ligne contient les paramètres obtenus par la
    fonction param de la n-ième image de la banque
    """

    A = []
    for num in S :
        start_intermediate_time = time.time()
        A.append(param_filtre(num))
        delta_intermediate = round(time.time() - start_intermediate_time)
        print("Image",num,", ",datetime.timedelta(seconds=delta_intermediate))

    return np.array(A)

## Tableau automatisation filtre

start_global_time = time.time()

N = 5000
S = range(1,N+1)
# S = range(701,1001)
# N_actuel = 8000
# N_new = 10000
# S = range(N_actuel+1,N_new+1)

arr_filtre = array_param_filtre(S)
df_filtre = pd.DataFrame(arr_filtre, index=S, columns=idx_filtre)
df_filtre.to_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N}, filtre).csv")
# df0 = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_actuel}).csv",index_col=0)
# df_join = pd.concat([df0,df])
# df_join.to_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_new}).csv")

delta_global = round(time.time() - start_global_time)
print("Temps global ",datetime.timedelta(seconds=delta_global))

## Tri en fonction du test de Kolmogorov

N_actuel = 5000
dff_filtre = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_actuel}, filtre).csv",index_col=0)

df_pvalue_filtre_005_1 = dff_filtre.loc[dff_filtre['pvalue1_filtre']>0.05,:].iloc[:,0:6]
df_pvalue_filtre_005_2 = dff_filtre.loc[dff_filtre['pvalue2_filtre']>0.05,:].iloc[:,0:6]
df_pvalue_filtre_001_1 = dff_filtre.loc[dff_filtre['pvalue1_filtre']>0.01,:].iloc[:,0:6]
df_pvalue_filtre_001_2 = dff_filtre.loc[dff_filtre['pvalue2_filtre']>0.01,:].iloc[:,0:6]

pvalues_filtre_005_1 = df_pvalue_filtre_005_1.index.tolist()
pvalues_filtre_005_2 = df_pvalue_filtre_005_2.index.tolist()
pvalues_filtre_001_1 = df_pvalue_filtre_001_1.index.tolist()
pvalues_filtre_001_2 = df_pvalue_filtre_001_2.index.tolist()

def tri_images(list,name):
    path_source='C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
    for i in list :
        img = recup_image(i,path_source)
        path_cible=f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Images_{name}/{i}.jpg"
        mpimg.imsave(path_cible,img)

tri_images(pvalues_filtre_005_1,'pvalues_filtre_005_1')
tri_images(pvalues_filtre_005_2,'pvalues_filtre_005_2')
tri_images(pvalues_filtre_001_1,'pvalues_filtre_001_1')
tri_images(pvalues_filtre_001_2,'pvalues_filtre_001_2')

## Impact de la variation de beta/sigma sur la pvalue, en diagonale

num = 1
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
img = recup_image(num,BD)
echelle = 1
WR = concatene(img,2,echelle)
T=np.reshape(WR,WR.shape[0]*WR.shape[1])
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
sigma_estim=sigma_chap(WR,beta_estim)
pval = stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim))[1]

taille_subdivision = 100
beta_vect = np.linspace(beta_estim-beta_estim/4,beta_estim+beta_estim/4,taille_subdivision).tolist()
sigma_vect = np.linspace(sigma_estim-sigma_estim/4,sigma_estim+sigma_estim/4,taille_subdivision).tolist()
vect = []
for i in range(0,taille_subdivision):
    vect.append([beta_vect[i],sigma_vect[i]])

start_global_time = time.time()

pval_vect = []
for beta,sigma in vect:
    pval_vect.append(stats.kstest(T, 'gennorm', (beta,0,sigma))[1])
print(max(pval_vect),pval)

ax = plt.axes(projection='3d')
ax.scatter3D(beta_vect, sigma_vect, pval_vect)
plt.show()

## Impact de la variation de beta/sigma sur la pvalue, en carré

num = 1
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
img = recup_image(num,BD)
echelle = 1
WR = concatene(img,2,echelle)
T=np.reshape(WR,WR.shape[0]*WR.shape[1])
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
sigma_estim=sigma_chap(WR,beta_estim)
pval = stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim))[1]

taille_subdivision = 10
zoom = 1/10
beta_vect = np.linspace(beta_estim-beta_estim*zoom,beta_estim+beta_estim*zoom,taille_subdivision).tolist()
sigma_vect = np.linspace(sigma_estim-sigma_estim*zoom,sigma_estim+sigma_estim*zoom,taille_subdivision).tolist()

pval_vect_carre = []
beta_axes = []
sigma_axes = []
start_global_time = time.time()
for beta in beta_vect:
    for sigma in sigma_vect:
        beta_axes.append(beta)
        sigma_axes.append(sigma)
        pval_vect_carre.append(stats.kstest(T, 'gennorm', (beta,0,sigma))[1])
print(max(pval_vect_carre))

delta_global = round(time.time() - start_global_time)
print("Temps global ",datetime.timedelta(seconds=delta_global))

ax = plt.axes(projection='3d')
ax.scatter3D(beta_axes, sigma_axes, pval_vect_carre)
ax.set_xlabel('beta')
ax.set_ylabel('sigma')
plt.title(f"pvalues pour l'image {num}, pvalue_initale={pval}")
plt.show()



# def pvalue_function(beta,sigma):
#     return stats.kstest(T, 'gennorm', (beta,0,sigma))[1]

# taille_subdivision = 10
# beta_vect = np.linspace(beta_estim-beta_estim/4,beta_estim+beta_estim/4,taille_subdivision)
# sigma_vect = np.linspace(sigma_estim-sigma_estim/4,sigma_estim+sigma_estim/4,taille_subdivision)
#
# beta_axe,sigma_axe = np.meshgrid(beta_vect,sigma_vect)
# pval_axe = [pvalue_function(beta,sigma) for beta,sigma in [beta_axe.tolist(),sigma_axe.tolist()]]
