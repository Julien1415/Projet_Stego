import skimage
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import numpy as np

from quantiles import quantiles_theoriques, quantiles_TCL
from images_sauv import liste_images
from PIL import Image, ImageFilter
from etude_coefs import recup_image
from test_wave import decomposition_image, concatene
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta


os.chdir("C:/Users/fseignat/Desktop/stega Seignat/code_images/ALASKA")

#Cov=recup_image(51610,"Cover")

##Bruit gaussien
# imageLue = Image.open("C:/Users/fseignat/Desktop/stega Seignat/code_images/ALASKA/Cover/00002.jpg")
# Pixel=np.asarray(imageLue)
# image_bruit=imageLue.filter(ImageFilter.GaussianBlur(radius = 3))
# Pixel_bruit=np.asarray(image_bruit)




## Compression
I=liste_images()


#Enregistre les images compressés 
for i in [2]:
    L=['0']*5
    S=list(str(i/10**4))
    S=np.delete(S,1)
    for i in range(len(S)):
        L[i]=S[i]
    s=''.join(L)
    che="Cover"+"/"+str(s)+'.jpg'
    imageLue = Image.open(che)
    imageLue.save("Compress_100/"+str(s)+".jpg", "jpeg", optimize = True, quality =20)
    






##Vérification fausses alarmes (compression)
nb_echelle=3
c=len(I)

# for i in I:
#     img=recup_image(i,"Cover")
#     taille=np.size(img)
#     img_compress=recup_image(i,"Compress_95")
#     Signe="Cover"
#     for e in range(nb_echelle):
#         WR=concatene(img,nb_echelle,e+1)
        
#         val_dep=val_dep_beta(WR)
#         beta_estim=beta_chap(WR,val_dep,10**(-4))
#         #sigma_estim=sigma_chap(WR,beta_estim)
#         if beta_estim!="Pas de solution":
#             den=np.sum(np.abs(WR)**beta_estim)

#             WR_Stego=concatene(img_compress,nb_echelle,e+1)
#             val_dep_ste=val_dep_beta(WR_Stego)
#             beta_estim_ste=beta_chap(WR_Stego,val_dep_ste,10**(-4))
#             #print(beta_estim,"  ",beta_estim_ste)
            
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
#         c-=1
#     print(Signe)

# print("\n")
# print(c/200)





##Vérification fausses alarmes (bruit gaussien)


# for i in I:
#     img=recup_image(i,"Cover")
#     taille=np.size(img)
#     PIL_img = Image.fromarray(np.uint8(img))
#     PIL_bruit=PIL_img.filter(ImageFilter.GaussianBlur(radius = 10))
#     img_bruit=np.asarray(PIL_bruit)
#     Signe="Cover"
#     for e in range(nb_echelle):
#         WR=concatene(img,nb_echelle,e+1)
        
#         val_dep=val_dep_beta(WR)
#         beta_estim=beta_chap(WR,val_dep,10**(-4))
#         #sigma_estim=sigma_chap(WR,beta_estim)
#         if beta_estim!="Pas de solution":
#             den=np.sum(np.abs(WR)**beta_estim)

#             WR_Stego=concatene(img_bruit,nb_echelle,e+1)
#             #val_dep_ste=val_dep_beta(WR_Stego)
#             #beta_estim_ste=beta_chap(WR_Stego,val_dep_ste,10**(-4))
#             #print(beta_estim,"  ",beta_estim_ste)
            
#             num=np.sum(np.abs(WR_Stego)**beta_estim)
#             rapport=num/den
#             #print(beta_estim,"  ",beta_estim_ste,"  ",rapport)
#             #Quantiles théoriques
#             #Q=quantiles_theoriques(taille,beta_estim,0.05)
            
            
            
#             #Quantiles TCL
#             Q=quantiles_TCL(np.size(WR),beta_estim,0.05)
#             if rapport>Q:
#                 Signe="Stego"
#     if Signe=="Stego":
#         c-=1
#     print(Signe)

# print("\n")
# print(c/200)