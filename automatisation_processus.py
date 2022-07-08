import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.image as mpimg
import time
import datetime
import os
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")    #chemin vers le dossier de travail

import wavelet2D
from fingerprinting import *
from test_wave import concatene, decomposition_image
from estimation_parametres import *


## Automatisation du traitement pour la banque Alaska

idx = ["beta_1", "sigma_1", "pvalue_1", "beta_2", "sigma_2", "pvalue_2"]    # liste des indices pour le dataframe pandas
for i in [k/10 for k in range(1,10)]:
    idx = idx + [f"beta_1 seuil={i}", f"sigma_1 seuil={i}", f"beta_2 seuil={i}", f"sigma_2 seuil={i}"]


def recuperation_image(num,path):
    """
    Parameters
    ----------
    num : si num=n, on récuperer la n-ieme image de la banques d'images, num>0
    path : chemin vers dossier contenant la banque

    Returns
    -------
    la matrice de l'image ainsi que le nom de l'image
    """

    list_files = [files for files in os.listdir(path) if os.path.isfile(os.path.join(path,files))]
    img_path = os.path.join(path,list_files[num-1])
    img = mpimg.imread(img_path)
    return img,list_files[num-1]


def param_alaska(num ,path = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'):    #ne fonctionne que pour la banque alaska
    """
    Parameters
    ----------
    num : numéro de l'image à traiter dans la banque
    path : chemin vers dossier contenant la banque alaska

    Returns
    -------
    renvoie différents parametres de l'image (notammment les sigma/beta
    des 2 premieres échelles, les sigma/beta de l'image modifiée en
    fonction du seuil et de la profondeur d'insertion...) associée au
    numéro, si elle existe
    """

    img = recup_image(num,path)
    L = []

    if type(img) == bool :
        return [np.nan]*len(idx)

    else :

        # paramètres sigma/beta et pvalue à l'échelle 1
        WR_1 = concatene(img,3,1)
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
        WR_2 = concatene(img,3,2)
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

        for i in [k/10 for k in range(1,10)]:

            # modification de l'image au seuil 0.1/0.9 avec un pas de 0.1 et à la profondeur 0
            clef = 0
            seuil = i
            img_mod = modif_image(img,clef,i)

            # paramètres sigma/beta à l'échelle 1 de l'image modifiée
            WR_mod_1 = concatene(img_mod,3,1)
            val_dep_mod_1 = val_dep_beta(WR_mod_1)
            beta_estim_mod_1 = beta_chap(WR_mod_1,val_dep_mod_1,10**(-4))
            sigma_estim_mod_1 = sigma_chap(WR_mod_1,beta_estim_mod_1)
            L = L + [round(beta_estim_mod_1,7), round(sigma_estim_mod_1,7)]

            # paramètres sigma/beta à l'échelle 2 de l'image modifiée
            WR_mod_2 = concatene(img_mod,3,2)
            val_dep_mod_2 = val_dep_beta(WR_mod_2)
            beta_estim_mod_2 = beta_chap(WR_mod_2,val_dep_mod_2,10**(-4))
            sigma_estim_mod_2 = sigma_chap(WR_mod_2,beta_estim_mod_2)
            L = L + [round(beta_estim_mod_2,7), round(sigma_estim_mod_2,7)]

        return L



def array_alaska(S, path = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'):
    """
    Parameters
    ----------
    S = liste des numéros des images à traiter
    path : chemin vers dossier contenant la banque

    Returns
    -------
    La matrice dont la n-ième ligne contient les paramètres obtenus par la
    fonction param de la n-ième image de la banque
    """

    A = []
    for num in tqdm(S) :
        # start_intermediate_time = time.time()
        A.append(param_alaska(num,path))
        # delta_intermediate = round(time.time() - start_intermediate_time)
        # print("Image",num,", ",datetime.timedelta(seconds=delta_intermediate))

    return np.array(A)


## Automatisation du traitement pour une banque quelconque, beta/sigma/pvalue,échelle 1/2,non filtrée puis filtrée à 5%

idx_filtre = ["nom", "beta1", "sigma1", "pvalue1", "beta2", "sigma2", "pvalue2", "beta1_filtre", "sigma1_filtre", "pvalue1_filtre", "beta2_filtre", "sigma2_filtre", "pvalue2_filtre"]    # liste des indices pour le dataframe pandas

def recuperation_image(num,path):
    """
    Parameters
    ----------
    num : si num=n, on récuperer la n-ieme image de la banques d'images, num>0
    path : chemin vers dossier contenant la banque

    Returns
    -------
    la matrice de l'image ainsi que le nom de l'image
    """

    list_files = [files for files in os.listdir(path) if os.path.isfile(os.path.join(path,files))]

    if num>len(list_files):
        return np.nan
    else :
        img_path = os.path.join(path,list_files[num-1])
        img = mpimg.imread(img_path)

    return img,list_files[num-1]

def param_qlq(num ,path = "C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso"):
    """
    Parameters
    ----------
    num : numéro de l'image dans la banque à traiter
    path : chemin vers dossier contenant la banque

    Returns
    -------
    renvoie beta/sigma/pvalue,échelle 1/2,non filtrée puis filtrée à 5% des images d'une banque qlq
    """

    img = recuperation_image(num,path)[0]
    L = [recuperation_image(num,path)[1]]
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

def array_qlq(S,path):
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
    for num in tqdm(S) :
        # start_intermediate_time = time.time()
        A.append(param_qlq(num,path))
        # delta_intermediate = round(time.time() - start_intermediate_time)
        # print("Image",num,", ",datetime.timedelta(seconds=delta_intermediate))

    return np.array(A)

## Creation tableau from Alaska

# start_global_time = time.time()

N_actuel = 8000
N_new = 10000
S = range(N_actuel+1,N_new+1)
S = range(1,10)

arr = array_alaska(S)
df = pd.DataFrame(arr, index=S, columns=idx)
# df0 = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_actuel}).csv",index_col=0)
# df_join = pd.concat([df0,df])
# df_join.to_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_new}).csv")

# delta_global = round(time.time() - start_global_time)
# print("Temps global ",datetime.timedelta(seconds=delta_global))

## Creation tableau from quelconque

start_global_time = time.time()

N_actuel = 32
N_new = 38
S = range(N_actuel+1,N_new+1)

path_banque = "C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso"

arr = array_qlq(S,path_banque)
df = pd.DataFrame(arr, index=S, columns=idx_filtre)
df.to_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau_banque_perso(1 to {N_new}).csv")
df0 = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau_banque_perso(1 to {N_actuel}).csv",index_col=0)
df_join = pd.concat([df0,df])
df_join.to_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau_banque_perso(1 to {N_new}).csv")

delta_global = round(time.time() - start_global_time)
print("Temps global ",datetime.timedelta(seconds=delta_global))

## Algo k-means

N_actuel = 10000
dfk = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_actuel}).csv",index_col=0)
Nb = 10000

# Cluster par k-mean sur beta_1/sigma_1
df_X1 = dfk[['beta_1','sigma_1']][0:Nb]
df_X1 = df_X1[~np.isnan(df_X1).any(axis=1)]      #remove the np.nan values from the df
X1 = df_X1.to_numpy()
kmeans_1 = KMeans(n_clusters=5).fit(X1)
Y_kmeans_1 = kmeans_1.predict(X1)
centers = kmeans_1.cluster_centers_
plt.scatter(X1[:, 0], X1[:, 1], c=Y_kmeans_1, s=8, cmap='viridis', alpha=0.8)
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.7)
plt.title(f"Clustering sur {Nb} images à l'échelle 1")
# plt.savefig(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Figures/Clustering sur {Nb} images à l'échelle 1.png")
# plt.show()
plt.close('all')

# X1_filtré = (limiter valeurs sigma)


# Cluster par k-mean sur beta_2/sigma_2
df_X2 = dfk[['beta_2','sigma_2']][0:Nb]
df_X2 = df_X2[~np.isnan(df_X2).any(axis=1)]
X2 = df_X2.to_numpy()
kmeans_2 = KMeans(n_clusters=5).fit(X2)
Y_kmeans_2 = kmeans_2.predict(X2)
centers = kmeans_2.cluster_centers_
plt.scatter(X2[:, 0], X2[:, 1], c=Y_kmeans_2, s=8, cmap='viridis', alpha=0.8)
plt.title(f"Clustering sur {Nb} images à l'échelle 2")
# plt.savefig(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Figures/Clustering sur {Nb} images à l'échelle 2.png")
# plt.show()
plt.close('all')

## Getting clusters df

df_cluster0_1 = df_X1[kmeans_1.labels_==0]
df_cluster1_1 = df_X1[kmeans_1.labels_==1]
df_cluster2_1 = df_X1[kmeans_1.labels_==2]
df_cluster3_1 = df_X1[kmeans_1.labels_==3]
df_cluster4_1 = df_X1[kmeans_1.labels_==4]

df_cluster0_2 = df_X2[kmeans_2.labels_==0]
df_cluster1_2 = df_X2[kmeans_2.labels_==1]
df_cluster2_2 = df_X2[kmeans_2.labels_==2]
df_cluster3_2 = df_X2[kmeans_2.labels_==3]
df_cluster4_2 = df_X2[kmeans_2.labels_==4]

## Filtrage sur le test de Kolmogorov

# N_actuel = 10000
dff = pd.read_csv(f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/csv/tableau(1 to {N_actuel}).csv",index_col=0)

df_pvalue_005_1 = dff.loc[dff['pvalue_1']>0.05,:].iloc[:,0:6]
df_pvalue_005_2 = dff.loc[dff['pvalue_2']>0.05,:].iloc[:,0:6]

df_pvalue_001_1 = dff.loc[dff['pvalue_1']>0.01,:].iloc[:,0:6]
df_pvalue_001_2 = dff.loc[dff['pvalue_2']>0.01,:].iloc[:,0:6]

## Tri des images

pvalues_005_1 = df_pvalue_005_1.index.tolist()
pvalues_005_2 = df_pvalue_005_2.index.tolist()

pvalues_001_1 = df_pvalue_001_1.index.tolist()
pvalues_001_2 = df_pvalue_001_2.index.tolist()

def tri_images(list,name):
    path_source='C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
    for i in list :
        img = recup_image(i,path_source)
        path_cible=f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Images_{name}/{i}.jpg"
        mpimg.imsave(path_cible,img)

tri_images(pvalues_005_1,'pvalues_005_1')
tri_images(pvalues_005_2,'pvalues_005_2')

tri_images(pvalues_001_1,'pvalues_001_1')
tri_images(pvalues_001_2,'pvalues_001_2')