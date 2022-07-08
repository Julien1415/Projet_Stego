import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import csv
import os
import random

os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")

from test_wave import decomposition_image, concatene
from GGD import densite_GGD
from estimation_parametres import sigma_chap, beta_chap,g_prime,val_dep_beta
from fingerprinting import modif_image
from scipy.stats import gennorm

import time
# start_time = time.time()


## Récupération des images
def recup_image(num,BD):
    '''
    
    Parameters
    ----------
    num : numéro de l'image à traiter
    BD : base de données (de type Cover ou JMiPOD)

    Returns
    -------
    renvoie l'image 

    '''
    L = ['0']*5
    S = list(str(num/10**4))
    S = np.delete(S,1)
    for i in range(len(S)):
        L[i] = S[i]
    s = ''.join(L)
    che = BD + "/" + str(s) + '.jpg'
    if os.path.exists(che):
        img = mpimg.imread(che)
    else:
        img = False                       #le chemin n'existe pas (vient du fait que la Base a des "trous", 00006.jpg n'existe pas par exemple)
    return img



def selection_images(N,BD):
    """
    
    Parameters
    ----------
    N : nombre d'images à récupérer
    BD : base de données (de type Cover ou JMiPOD)

    Returns
    -------
    renvoie une liste de N nombres représentant les images à sélectionner
    (le problème est de sélectionner une seule image en faisant attention aux "trous" de la banque Alaska )

    """
    A=random.sample(range(80006),80006)
    L=[]
    i=0
    j=0
    while j<N:
        if type(recup_image(A[i],"Cover"))!=bool:
            L.append(A[i])
            j+=1
        i+=1
    return L

## Image naturelle (Cover) pour estimer les paramètres

BD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Banques_alaska' #chemin vers dossier images
# img = mpimg.imread(BD+'/00001.jpg')
img = recup_image(1,BD)


# clef=0
# seuil=0
# img=modif_image(img,clef,seuil)
# plt.imshow(img)


#Coefficients d'ondelettes en fonction du nombre d'échelles, de l'échelle et du plan
nb_echelle=3
echelle=3
nb_plan=3
plan=1
# WR=concatene(img,nb_echelle,echelle)
# print(WR)



#Estimation paramètres
# val_dep=val_dep_beta(WR)
# beta_estim=beta_chap(WR,val_dep,10**(-4))
# sigma_estim=sigma_chap(WR,beta_estim)

# print(' val_dep=',val_dep, '\n',' beta_estim = ',beta_estim ,'\n','sigma_estim = ',sigma_estim)
# print("--- %s seconds ---" % (time.time() - start_time))




##Rassemblement des informations dans des tableaux


# nb_images=21
# Seuil=np.arange(0,1,0.1)

# with open('data_COVER.csv','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Nom', 'Echelle', 'Plan', 'Beta_Chapeau','Sigma_chapeau'])
#       writer.writerow('\n')
#       for j in range(0,nb_images):
#           img=recup_image(j)
#           print("j=",j)
#           if type(img)!=bool:                                #image existe(cela renvoie bien un tableau)
#                 #for s in Seuil:
#                     #s=round(s,1)
#                     #img=modif_image(img,clef,s)
#                     for k in range(0,nb_plan):
#                         for i in range(0,nb_echelle):
#                             WR=decomposition_image(img,nb_echelle,i+1,k+1)
#                             val_dep=val_dep_beta(WR)
#                             beta_estim=beta_chap(WR,val_dep,10**(-4))
#                             sigma_estim=sigma_chap(WR,beta_estim)
#                             writer.writerow([str(j)+'.jpg', str(i+1), str(k+1),str(beta_estim),str(sigma_estim)])
#                         writer.writerow('\n')
#           writer.writerow('\n')
#           writer.writerow('\n')



##Moyenne pour une image (typiquement 500 simulations)

Seuil=np.arange(0,1,0.1)
#S=selection_images(10,"Cover")
#S=[4913, 64164, 65768, 22590, 2627, 8563, 15716, 2139, 34515, 16586]


##récupération des beta des images Cover

# B_Cov=[]
# S_Cov= []

# for images in S:
#     I=recup_image(images,"Cover")
#     for i in range(nb_echelle):
#         WR=concatene(I,nb_echelle,i+1)
#         val_dep=val_dep_beta(WR)
#         beta_est=beta_chap(WR,val_dep,10**(-4))
#         sigma_est=sigma_chap(WR,beta_est)
#         B_Cov.append(beta_est)
#         S_Cov.append(sigma_est)
#print(B_Cov)
#print("\n",S_Cov)


B_Cov=[1.1515723964180615,
 1.6962770121229283,
 1.1452405198169708,
 0.8344192087790803,
 1.0305013596333106,
 0.5934323249867881,
 1.2289213733226507,
 0.3877500172660005,
 0.2617640944531531,
 0.7645901315927348,
 0.9439178995554338,
 0.9807625013117248,
 0.5928469382879877,
 0.5012033451951253,
 0.4276422080811947,
 0.4944430317903056,
 0.5468057222324614,
 0.537978999683653,
 0.77991087477478,
 1.1057574007248554,
 0.9635635333165171,
 0.8342620560408048,
 1.4925210486542484,
 1.0191832039104112,
 0.6355490723118158,
 0.579818047093477,
 0.558967162612578,
 0.6702321249793408,
 0.7246492509603742,
 0.8548908753279673]

S_Cov=[2.4359452746046864,
       7.69279590624291,
       7.994011017422969,
       5.119409285152012,
 8.483090026092723,
 1.9505377639214478,
 5.282274501850235,
 0.06463208551202543,
 0.0029567157706211823,
 0.63062194767279,
 4.890124090940542,
 17.244168235872156,
 0.20373837732593642,
 0.25813895210155646,
 0.2952271066303459,
 0.3064438805095296,
 1.7995197293160319,
 4.827005887665468,
 0.5442929333954132,
 4.239465506242737,
 10.038411063610281,
 0.6371321022507558,
 5.534898470933981,
 4.842562083742451,
 1.5118013641481733,
 1.707042792325919,
 2.402925005063707,
 0.6982801749863614,
 3.3736545268149154,
 17.300317178038476]


## Permet de calculer le sigma en prenant beta fixe

# with open('coefs_3bits.csv','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Image','Echelle','Seuil', 'Beta_Chapeau','Sigma_chapeau'])
#       writer.writerow('\n')
#       z=0
#       for c in range(len(S)):
#         print("c=",S[c])
#         img=recup_image(S[c],"Cover")
#         for i in range(0,nb_echelle):
#             #beta_estim=B_Cov[c*3+i]
#             for s in Seuil:
#                 z+=1
#                 print(z)
#                 s=round(s,1)
#                 for l in range(500):
#                           Si=[]
#                           B=[]
#                           img_mod=modif_image(img,0,s)
#                           img_mod=modif_image(img_mod,1,s)
#                           img_mod=modif_image(img_mod,2,s)
#                           WR=concatene(img_mod,nb_echelle,i+1)
#                           val_dep=val_dep_beta(WR)
#                           beta_est=beta_chap(WR,val_dep,10**(-4))
#                           sigma_estim=sigma_chap(WR,beta_est)
#                           Si.append(sigma_estim)
#                           B.append(beta_est)
#                 sig=np.mean(Si)
#                 beta=np.mean(B)
#                 writer.writerow([str(S[c]),str(i+1),str(s),str(beta),str(sig)])
#         writer.writerow('\n')


# Taux=np.arange(0.1,1.1,0.1)

# with open('Param_Substi','a',newline='') as fichiercsv:
#       writer=csv.writer(fichiercsv)
#       writer.writerow(['Image','Echelle','Taux d\'insertion', 'Beta_Chapeau','Sigma_chapeau'])
#       writer.writerow('\n')
#       z=0
#       for c in S:
#         print("c=",c)
#         img=recup_image(c,"Cover")
#         for i in range(0,nb_echelle):
#             for t in Taux:
#                 z+=1
#                 print(z)
#                 t=round(t,1) 
#                 for l in range(1):
#                           B=[]
#                           S=[]
#                           #img_mod=modif_image(img,0,s)
#                           #img_mod=modif_image(img_mod,1,s)
#                           #img_mod=modif_image(img_mod,2,s)
#                           M=message(512**2*t)
#                           img_mod=modif_LSB(img, M)
#                           WR=concatene(img_mod,nb_echelle,i+1)
#                           val_dep=val_dep_beta(WR)
                      
#                           beta_estim=beta_chap(WR,val_dep,10**(-4))
#                           B.append(beta_estim)
#                           sigma_estim=sigma_chap(WR,beta_estim)
#                           S.append(sigma_estim)
#                 bet=np.mean(B)
#                 sig=np.mean(S)
#                 writer.writerow([str(c),str(i+1),str(t),str(bet),str(sig)])
#         writer.writerow('\n')








#Tracé des différents paramètres 
# Cov=open('COVER.csv','r')
# r_cov = csv.reader(Cov, delimiter=',')
# liste_cov = list(r_cov)
# Cov.close()


# file=open('Stego_3bits.csv','r')
# r = csv.reader(file, delimiter=',')
# liste = list(r)
# file.close()

# file=open('Param_Substi.csv','r')
# r = csv.reader(file, delimiter=',')
# liste = list(r)
# file.close()

# # #    Séléction pour chaque image
# IM1=liste[2:32]
# Cov1=S_Cov[0:3]

# IM2=liste[33:63]
# Cov2=S_Cov[3:6]

# IM3=liste[64:94]
# Cov3=S_Cov[6:9]

# IM4=liste[95:125]
# Cov4=S_Cov[9:12]

# IM5=liste[126:156]
# Cov5=S_Cov[12:15]

# IM6=liste[157:187]
# Cov6=S_Cov[15:18]

# IM7=liste[188:218]
# Cov7=S_Cov[21:24]

# IM8=liste[219:249]
# Cov8=S_Cov[24:24]

# IM9=liste[250:280]
# Cov9=S_Cov[24:27]

# IM10=liste[281:311]
# Cov10=S_Cov[27:30]


#     #Tracer les graphes 
    
    

    
# def evolution_parametre_insertion(Cov,IM,echelle,nb):
#     """
    

#     Parameters
#     ----------
#     Cov: paramètre de l'image cover
#     IM : prend une liste représentant les coefficients pour une image retouchée
#     echelle : echelle à étudier
#     nb :indice de l'image dans la liste S

#     Returns
#     -------
#     renvoie 3 graphiques (car 3 échelles) représentant l'évolution des paramètres en fonction du seuil ainsi que la valeur théorique

#     """
#     Selec_modif=np.array(IM[(echelle-1)*10:echelle*10])           #données de l'image modifiée
#     Selec_Cov=np.array(Cov[echelle-1])
#     Seuil=np.arange(0,1,0.1)
#     #Selection paramètres
#     Sigma_modif=list(map(float,Selec_modif[:,4]))       #permet le conversion en float
#     #Beta_modif=list(map(float,Selec_modif[:,3]))
#     Sigma_Cover=np.ones(len(Seuil))*float(Cov[echelle-1])
#     #Beta_Cover=np.ones(len(Seuil))*B_Cov[3*(nb-1)+echelle-1]
    
    
#     #Tracé
#     fig, ax1 = plt.subplots()

#     ax2 = ax1.twinx()
#     p1,=ax1.plot(Seuil,Sigma_modif, 'b:o',label="Sigma modifié")
#     p2,=ax1.plot(Seuil, Sigma_Cover, 'b',label="Sigma Cover")
#     plt.legend()
    
#     #p3,=ax2.plot(Seuil,Beta_modif, 'y:o',label='Beta modifié')
#     #p4,=ax2.plot(Seuil,Beta_Cover, 'y',label='Beta_Cover')

#     ax1.set_xlabel('Seuil',fontsize=10)
#     ax1.set_ylabel('Sigma', color='black',fontsize=10)
#     ax2.set_ylabel('Beta', color='black',fontsize=10)
    
#     nom=IM[0][0]

#     plt.title("Image "+str(nom)+" à l'échelle " +str(echelle),fontsize=10)
#     ax1.legend(handles=[p1, p2, ],fontsize=7)
#     plt.show()
#     return()



# for echelle in [1,2,3]:
#   evolution_parametre_insertion(Cov2,IM2,echelle,1)






#Vérification de la loi GGd pour les coefficients d'ondelettes

# img=recup_image(3,'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759')
# 
# # # WR=decomposition_image(img,nb_echelle,0,0)
# # # plt.imshow(WR)
# # # WR_mod=decomposition_image(img_mod,nb_echelle,1,1)
# WR=concatene(img,2,1)
# val_dep=val_dep_beta(WR)
# beta_estim=beta_chap(WR,val_dep,10**(-4))
# sigma_estim=sigma_chap(WR,beta_estim)
# 
# plt.figure()
# plt.hist(np.reshape(WR,WR.shape[0]*WR.shape[1]), bins='auto',color='red',density=True, alpha=0.3, rwidth=0.85)
# 
# x = np.linspace(np.min(WR),np.max(WR),150)
# plt.plot(x, gennorm.pdf(x,beta_estim,scale=sigma_estim),'r-', lw=2,color='blue', alpha=0.2, label='gennorm pdf')
# plt.ylabel("Densité")
# plt.ylim([0,0.06])
# plt.title("Histogramme des coefficients d'ondelettes et densité de la GGD en bleu")
# plt.show()
