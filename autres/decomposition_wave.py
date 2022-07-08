import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

from GGD import densite_GGD

# Load image
#original = pywt.data.camera()
img=mpimg.imread('stego_2.jpg')
R=np.array(img[:,:,0],'float')

G=np.array(img[:,:,1],'float')

B=np.array(img[:,:,2],'float')

original=R+G+B

# Wavelet transform of image, and plot approximation and details with Haar
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.wavedec2(original, 'haar',level=1)
ca2, sli2 = pywt.coeffs_to_array(coeffs2)
LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()


### Daubechies 4
coeffs = pywt.wavedec2(original, 'db4',level=3)
ca, sli = pywt.coeffs_to_array(coeffs)
cA3, (cH3, cV3, cD3),(cH2, cV2, cD2),(cH1, cV1, cD1) = coeffs
plt.hist(cD3/100,density=True)
#densite_GGD(10000,0.055,0.46)


##Visualisation du tableau
# plt.figure(10)
# plt.imshow(ca, cmap="gray")
# plt.colorbar()
# plt.title('Coefs approx + details')
# plt.show()

# plt.figure(11)
# plt.imshow(cA3, cmap="gray")
# plt.colorbar()
# plt.title('Coeff. approximation')
# plt.show()

# plt.figure(12)
# #plt.plot(131)
# #plt.subplot(131)
# plt.imshow(cH3, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cH3,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details horizontaux niveau 3',fontsize=16,loc='right')
# #plt.subplot(132)
# plt.figure(13)
# plt.imshow(cV3, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cV3,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.title('Details verticaux niveau 3',fontsize=16,loc='right')
# #plt.subplot(133)
# plt.figure(14)
# plt.imshow(cD3, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cD3,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details diagonaux niveau 3',fontsize=16,loc='right')



# plt.figure(15)
# plt.imshow(cH2, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cH2,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details horizontaux niveau 2',fontsize=16,loc='right')

# plt.figure(16)
# plt.imshow(cV2, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cV2,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details verticaux niveau 2',fontsize=16,loc='right')
# plt.figure(17)
# plt.imshow(cD2, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cD2,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details diagonaux niveau 2',fontsize=16,loc='right')
# plt.show()

# plt.figure(18)

# plt.imshow(cH1, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cH1,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details horizontaux niveau 1',fontsize=16,loc='right')
# plt.figure(19)
# plt.imshow(cV1, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cV1,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details verticaux niveau 1',fontsize=16,loc='right')
# plt.figure(20)
# plt.imshow(cD1, cmap="gray")
# ax = plt.gca()
# im = ax.imshow(cD1,cmap="gray")

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)
# plt.title('Details diagonaux niveau 1',fontsize=16,loc='right')
# plt.show()