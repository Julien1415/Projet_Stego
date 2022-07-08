import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gennorm
import matplotlib.image as mpimg
from PIL import Image
import os
os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")    #chemin vers le dossier de travail

from fingerprinting import *
from test_wave import concatene, decomposition_image
from estimation_parametres import *

## Insertion
num = 8
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
img = recup_image(num,BD)
# plt.imsave("C:/Users/Julie/OneDrive/Documents\Stage_M2/Travail/Figures/image_8.jpg",img)
# plt.imshow(img)
# plt.show()

clef = 0
seuil = 0.3
img_mod = modif_image(img,clef,seuil)
plt.imshow(img_mod)
plt.show()
# plt.imsave(f"C:/Users/Julie/OneDrive/Documents\Stage_M2/Travail/Figures/image_{num}_bit{clef}_seuil{seuil}.jpg",img_mod)

##
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
list_Cover = [files for files in os.listdir(BD) if os.path.isfile(os.path.join(BD,files))]
img_path = os.path.join(path_source,list_Cover[0])
img = mpimg.imread(img_path)
img = modif_image(img,0,seuil)
plt.imshow(img)
plt.show()

## Creation d'un ensemble d'images modifiées par fingerprinting
nbre_images = 1000

path_source = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
seuil = 0.3
list_Cover = [files for files in os.listdir(path_source) if os.path.isfile(os.path.join(path_source,files))]
for i in range(nbre_images):
    img_path = os.path.join(path_source,list_Cover[i])
    # print(img_path)
    img = mpimg.imread(img_path)
    img = modif_image(img,0,seuil)
    path_cible = f"C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Fingerprinted_seuil0.3/{i}.jpg"
    mpimg.imsave(path_cible,img)

## Fit entre hist et GGD
num = 8422
num = 3
BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
img = recup_image(num,BD)
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/homme-avec-un-appareil.jpg')
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/lena.jpeg')
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/lena_gray.jpeg')
# img = np.asarray(Image.open('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/kodim18.png'))
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/rocks.jpg')
# img = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Banque_perso/sgrA.jpg')

echelle = 1
WR = concatene(img,echelle,echelle)
val_dep=val_dep_beta(WR)
beta_estim=beta_chap(WR,val_dep,10**(-4))
beta_round=round(beta_estim,4)
sigma_estim=sigma_chap(WR,beta_estim)
sigma_round=round(sigma_estim,4)
T=np.reshape(WR,WR.shape[0]*WR.shape[1])
T_tri = np.sort(T)
taille_T = len(T_tri)

pval = stats.kstest(T_tri, 'gennorm', (beta_estim,0,sigma_estim))[1]
# print(stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim)))


plt.figure()
plt.hist(T, bins='auto',density=True, color='red', alpha=0.4, rwidth=1)
x = np.linspace(np.min(WR),np.max(WR),1000)
plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'b-', lw=1.5, alpha=0.35)
plt.title(f"Cover {num}, beta = {beta_round}, sigma = {sigma_round}")
plt.xlim([-250,250])
# plt.xlabel(f"pvalue={pval}")
plt.show()
plt.close('all')

## Problot
ech_ggd = gennorm.rvs(beta_estim, scale=sigma_estim, size=taille_T)
ech_ggd_tri = np.sort(ech_ggd)
plt.figure()
stats.probplot(ech_ggd,dist='gennorm',sparams=(beta_estim,0,sigma_estim),plot=plt)
plt.figure()
plt.scatter(T_tri,ech_ggd_tri,c='k',s=6,alpha=0.5)
plt.plot(T_tri,T_tri,'r')
plt.title(f"Coeffs contre échantillon GGD pour l'image {num}, pval={round(pval,10)}")
plt.xlabel("Coeffs d'ondelettes")
plt.ylabel("Echantillon GGD")
plt.show()

## Impact du changement de bits sur les parametres
clef = 0
seuil = 0.3
img_mod = modif_image(img,clef,seuil)

echelle = 1
WR_mod = concatene(img_mod,echelle,echelle)
# print(WR_mod)
# print(WR_mod.shape)
T_mod=np.reshape(WR_mod,WR_mod.shape[0]*WR_mod.shape[1])

val_dep_mod=val_dep_beta(WR_mod)
beta_estim_mod=beta_chap(WR_mod,val_dep,10**(-4))
# print("beta_mod = ",beta_estim_mod)
beta_round_mod=round(beta_estim_mod,4)
sigma_estim_mod=sigma_chap(WR_mod,beta_estim)
# print("sigma_mod = ",sigma_estim_mod)
sigma_round_mod=round(sigma_estim_mod,4)

plt.figure()
plt.hist(T_mod, bins='auto', density=True, color='red', alpha=0.4)
x = np.linspace(np.min(WR_mod),np.max(WR_mod),100)
plt.plot(x, gennorm.pdf(x,beta_estim_mod,scale=sigma_estim_mod),'b-', lw=1.5, alpha=0.35)
plt.title(f"beta_mod = {beta_round_mod}, sigma_mod = {sigma_round_mod}, clef = {clef}, seuil = {seuil}")
plt.xlim([-250,250])
plt.show()
plt.close('all')

## Test d'adéquation sur echantillon GGD
beta = 0.5
sigma = 10
ech_gg = gennorm.rvs(beta, scale=sigma, size=200000)

pval_ech = stats.kstest(ech_gg, 'gennorm',(0.5,0,10))[1]

plt.figure()
plt.hist(ech_gg, bins='auto', density=True, color='red', alpha=0.4)
x = np.linspace(np.min(ech_gg),np.max(ech_gg),150)
plt.plot(x, gennorm.pdf(x,beta,scale=sigma),'b-', lw=1.5, alpha=0.5)
plt.title(f"beta={beta}, sigma={sigma}, pvalue={round(pval_ech,5)}")
plt.show()
plt.close('all')

## Tests sur l'histogramme
# T_norm=T/sum(T)

# plt.figure()
# h = plt.hist(T, bins='auto', color='red', alpha=0.5)
# plt.plot(T)
# plt.plot(h[1])
# plt.show()

# h_norm = h[1]/sum(h[1])


# stats.kstest(T, gennorm.pdf(x,beta_estim,scale=sigma_estim))
# stats.kstest(h_norm, gennorm.pdf(x,beta_estim_mod,scale=sigma_estim_mod))

## Test fit entre les parametres estimés et gennorm.fit
beta0 = gennorm.fit(T)[0]
loc0 = gennorm.fit(T)[1]
sigma0 = gennorm.fit(T)[2]

# plt.figure()
# plt.hist(T, bins='auto',density=True, color='red', alpha=0.5)
# x = np.linspace(np.min(WR),np.max(WR),150)
# plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'b-', lw=1.5, alpha=0.3)
# plt.plot(x, gennorm.pdf(x,beta0,scale=sigma0),'g-', lw=1.5, alpha=0.3)
# plt.show()

print((beta_estim,0,sigma_estim))     # les parametres sont tres proches ce qui est rassurant quant à la qualité de l'estimation des parametres
print(gennorm.fit(T))

print(stats.kstest(T, 'gennorm', (beta_estim,0,sigma_estim)))
print(stats.kstest(T, 'gennorm', (gennorm.fit(T))))

## Test comparativement à un échantillon GGD
ech_ggd = gennorm.rvs(beta_estim, scale=sigma_estim, size=taille_T)
ech_ggd0 = gennorm.rvs(beta0,loc = loc0, scale=sigma0, size=taille_T)
print(stats.kstest(T,ech_ggd))
print(stats.kstest(T,ech_ggd0))