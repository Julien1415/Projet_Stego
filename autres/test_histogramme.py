import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gennorm
from scipy.stats import norm
import matplotlib.image as mpimg
import time
import os

os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")

import wavelet2D
from fingerprinting import *
from test_wave import concatene, decomposition_image
from estimation_parametres import *
from etude_coefs import recup_image

## Tests divers

# img_1 = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travaux/Images/stinkbug.png')
# img_2 = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Travaux/Images/ladybug.jpeg')
# wc = mpimg.imread('C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Banques_alaska/00001.jpg')

# img_b=wc[:,:,2]
# imgplot = plt.imshow(wc)
# plt.show()
# imgplot = plt.imshow(img_b)
# plt.show()

# img=Lire_Echelle_Mallat2D(img_2,2,3)
# img_m=Ecrire_Echelle_Mallat2D(img_2,img,2,3)
# imgplot = plt.imshow(img_m)
# plt.show()

# plt.plot(makeonfilter('Daubechies',4))
# plt.show()

# h0=makeonfilter('Daubechies',4)
# print(banc_filtre2d(0,wc,1,h0))

# n=wc.shape[0]
# m=wc.shape[1]
# L=2
# copy=np.copy(wc[0:int(n/(2**L))*(2**L),0:int(m/(2**L))*(2**L)])
# imgplot = plt.imshow(copy)
# plt.show()

## Test d'adéquation sur échantillon loi normale

ech_norm = stats.norm.rvs(loc=2,scale=5.2,size=200000)

# plt.figure()
# plt.plot(ech_norm)
# plt.figure()
# plt.hist(ech_norm, bins='auto', density=True, color='red', alpha=0.5)
x = np.linspace(np.min(ech_norm),np.max(ech_norm),500)
# plt.plot(x, stats.norm.pdf(x),'b-', lw=2, alpha=0.3)
# plt.show()
# plt.close('all')

print(stats.kstest(ech_norm, 'norm',(2,5)))
# print(stats.kstest(ech_norm, norm.cdf(2,5))
# print(stats.kstest(ech_norm, norm.cdf(x)))

## Densité loi normale/Loi GGD

plt.figure()
x = np.linspace(np.min(ech_norm),np.max(ech_norm),150)
plt.plot(x, stats.norm.pdf(x),'r-', lw=1., alpha=0.3)
plt.plot(x, gennorm.pdf(x,2,scale=1/np.sqrt(2)),'b-', lw=1.5, alpha=0.3)    # donc scale=sigma (documentation fausse)
plt.show()
