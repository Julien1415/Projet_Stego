import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import time

os.chdir("C:/Users/fseignat/Desktop/stega Seignat/code_images")

from quantiles import quantiles_theoriques,quantiles_TCL
from lecture_video import tableau,cv_plt,distance,recup_Cover,modif_video, supp_val_extrem
from test_wave import decomposition_image, concatene
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta
from scipy.stats import gennorm
from etude_coefs import recup_image


start_time = time.time()


##Recuperation des fichiers

#KB = np.load('pixel_video_KB.npy')
#TR = np.load('pixel_video_TR.npy')
#TR2 = np.load('pixel_video_TR(2).npy')
#SC=np.load("Scarface.npy")
#AZ=np.load("Azur et Asmar.npy")
BA_TR=np.load("BA_TR.npy")

#Vidéo non modifiée
file=BA_TR
File=[file]

Nom_File=["KB","TR","TR2","SC","AZ"]
#File=[KB,TR,TR2,SC,AZ]



nb_echelle=3

# #NB=len(file)


##♥ Analyse d'une vidéo entière 

# Count_Ste=[]


# f=0
# for file in File:
#     c=0
#     Rapp_sigma=[]
#     Rapp_beta=[]
#     NB=len(file)
#     for ind in range(1,NB-1):
#         Signe="Cover"
#         for e in range(1):
        
#             img=file[ind]
        
#             ind_Cover=recup_Cover(file,ind)
#             img_Cover=file[ind_Cover]
        
#             WR=concatene(img,nb_echelle,e+1)
#             WR_C=concatene(img_Cover,nb_echelle,e+1)
            
#             WR=supp_val_extrem(WR,0.002)
#             WR_C=supp_val_extrem(WR_C,0.002)

        
#             #Image à tester
#             val_dep=val_dep_beta(WR)
#             beta_estim=beta_chap(WR,val_dep,10**(-4))
#             sigma_estim=sigma_chap(WR,beta_estim)
        
#             #Image "Cover"
#             val_dep_C=val_dep_beta(WR_C)
#             beta_estim_C=beta_chap(WR_C,val_dep_C,10**(-4))
#             sigma_estim_C=sigma_chap(WR_C,beta_estim_C)
                
#             if beta_estim!="Pas de solution":
#                     num=np.sum(np.abs(WR)**beta_estim)
#                     den=np.sum(np.abs(WR_C)**beta_estim)                
#                     rapport=num/den
                    
                    
#                     #Quantiles théoriques
#                     Q_th=quantiles_theoriques(np.size(WR_C),beta_estim,0.05)
                    
                    
#                     #Quantiles TCL
#                     #Q_TCL=quantiles_TCL(np.size(WR_Cov),beta_estim,0.05)
                   
                    
#                     if rapport>Q_th:
#                         Signe="Stego"
                        
#             #print("ind = ", ind,"beta_Cov = ",beta_estim_C,"beta_test =",beta_estim)       
#             #print("\n")
#             if Signe=="Stego":
#                     c+=1
#         if beta_estim!='Pas de solution' and beta_estim_C !='Pas de solution':
#             Rapp_sigma.append(sigma_estim/sigma_estim_C)
#             Rapp_beta.append(beta_estim/beta_estim_C)
#         print("F= ",f,"ind =",ind,Signe)
#         print("\n")
    
#     print("\n")
#     Count_Ste.append(c)
    
    
    
#     B=[i for i in range(len(Rapp_beta))]
    
#     plt.figure()
#     plt.plot(B,Rapp_beta)
#     plt.title(Nom_File[f]+"_Rapport des beta"+"_2")      #entre l'image test et l'image désignée Cover (par la norme L²)
    
#     plt.figure()
#     plt.plot(B,Rapp_sigma)
#     plt.title(Nom_File[f]+"_Rapport des sigma"+"_2") 
#     f+=1


# print("--- %s seconds ---" % (time.time() - start_time))


## Analyse d'une trame (pour comprendre le comportement d'une image spécifique)

img=file[187]              #154

ind_Cover=recup_Cover(file,187)
img_Cover=file[ind_Cover]

#img=recup_image(7841,"C:/Users/fseignat/Desktop/stega Seignat/code_images/ALASKA/Cover")
WR=concatene(img,2,2)
WR_C=concatene(img_Cover,2,2)



#Seuillage
WR=supp_val_extrem(WR,0.002)
WR_C=supp_val_extrem(WR_C,0.002)




#WR=decomposition_image(img,3,2,1)
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
sigma_estim=sigma_chap(WR,beta_estim)
print("beta = ",beta_estim,"sigma =",sigma_estim,"val =",val_dep)

val_dep_C=val_dep_beta(WR_C)
beta_estim_C=beta_chap(WR_C,val_dep_C,10**(-4))
sigma_estim_C=sigma_chap(WR_C,beta_estim_C)
print("beta = ",beta_estim_C,"sigma =",sigma_estim_C,"val =",val_dep_C)


plt.figure()
n, b, patches=plt.hist(np.reshape(WR,np.size(WR)), bins='sqrt',color='blue',density=True, alpha=0.3, rwidth=0.85)
x = np.linspace(np.min(WR),np.max(WR),150)
F=gennorm.pdf(x,beta_estim,scale=sigma_estim)
plt.plot(x,F,'r-', lw=1, alpha=0.6, label='gennorm pdf')
plt.ylim([0,np.max(F)])
plt.xlim([0,n.max()])


plt.figure()
plt.hist(np.reshape(WR_C,np.size(WR_C)), bins='sqrt',color='blue',density=True, alpha=0.3, rwidth=0.85)
x_C = np.linspace(np.min(WR_C),np.max(WR_C),150)
F_C=gennorm.pdf(x_C,beta_estim_C,scale=sigma_estim_C)
plt.plot(x_C,F_C,'r-', lw=1, alpha=0.6, label='gennorm pdf')
plt.ylim([0,np.max(F_C)])





# ##Dessins animés

# # AZ=np.load("Azur et Asmar.npy")

# # for i in range(10):
# #     img=AZ[i]
# #     for e in range(nb_echelle):
# #         WR=concatene(img,nb_echelle,e+1)
# #         val_dep=val_dep_beta(WR)
# #         beta_estim=beta_chap(WR,val_dep,10**(-4))
# #         sigma_estim=sigma_chap(WR,beta_estim)
# #         print("beta =", beta_estim, "sigma =",sigma_estim)
# #         plt.figure()
# #         plt.hist(np.reshape(WR,WR.shape[0]*WR.shape[1]), bins='auto',color='blue',density=True, alpha=0.5, rwidth=0.85)
# #         x = np.linspace(gennorm.ppf(0.001,beta_estim,scale=sigma_estim),gennorm.ppf(0.999, beta_estim,scale=sigma_estim), 100)
# #         F=gennorm.pdf(x,beta_estim,scale=sigma_estim)
# #         plt.plot(x, F,'r-', lw=1, alpha=0.5,color='red', label='gennorm pdf')
# #         plt.ylim([0,np.max(F)])


