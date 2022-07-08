import numpy as np
import math
import numpy.random as npr
import matplotlib.pyplot as plt

from scipy.stats import gennorm


def PDF_GGD(x,sigma,beta):
    """

    Calcule la densité de probabilité de la loi gaussienne généralisée (centrée)

    """
    f=beta/(2*sigma*math.gamma(1/beta))*np.exp(-(np.abs(x)/sigma)**beta)
    return f



def densite_GGD(N,sigma,beta):
    """

    Parameters
    ----------
    N : nombre de données conservées via Monte-Carlo
    sigma,beta : paramètre de la GGD

    Returns
    -------
    renvoie la densité de la GDD

    """
    c=0
    L=[]
    while c<N:
        x=-100+200*npr.random(1)     #je considère que f(x)=0 pour |x|>300
        y=npr.random(1)
        if PDF_GGD(x, sigma, beta)>y:
            c+=1
            L.append(float(x))
    plt.hist(L,bins='auto',density=True,alpha=0.9,color='green')
    return L


# L=densite_GGD(30000,1,5)

# #Permet de tracer automatiquement
# fig, ax = plt.subplots(1, 1)
# beta=1
# x = np.linspace(gennorm.ppf(0.01,beta,scale=5),gennorm.ppf(0.99, beta,scale=5), 100)
# ax.plot(x, gennorm.pdf(x,beta,scale=5),'r-', lw=5, alpha=0.6, label='gennorm pdf')

# sigma = 4
# beta = 0.5
# plt.figure()
# x=np.linspace(-10,10,100)
# plt.plot(x, PDF_GGD(x,sigma,beta),'r-',alpha=0.3)
# # plt.figure()
# plt.plot(x, gennorm.pdf(x,beta,scale=sigma),'b-',alpha=0.3)
# plt.show()
# plt.close('all')