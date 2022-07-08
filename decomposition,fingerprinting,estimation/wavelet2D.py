import numpy as np
import matplotlib.image as mpimg


def Lire_Echelle_Mallat2D(wc,l,ind): #wc=image d'entrée,l=echelle,ind=localisation du rectangle(indice)
    
    n = wc.shape[0]
    m = wc.shape[1]
    if ind==1:
        img=wc[0:int(n/(2**l)),int(m/(2**l)):int(m/(2**(l-1)))] #récupère les pixels du rectangle superieur droit à l'echelle l
    if ind==2:
        img=wc[int(n/(2**l)):int(n/(2**(l-1))),0:int(m/(2**l))] #récupère les pixels du rectangle inférieur gauche à l'echelle l
    if ind==3:
        img=wc[int(n/(2**l)):int(n/(2**(l-1))),int(m/(2**l)):int(m/(2**(l-1)))] #récupère les pixels du rectangle inférieur droit à l'echelle l
    if ind==0:
        img=wc[0:int(n/(2**(l))),0:int(m/(2**(l)))] #récupère les pixels du rectangle superieur gauche à l'echelle l
    return img

def Ecrire_Echelle_Mallat2D(wc,img,l,ind): #remplace un rectangle de l'image wc
    
    n = wc.shape[0]
    m = wc.shape[1]
    if ind==1:
        wc[0:int(n/(2**l)),int(m/(2**l)):int(m/(2**(l-1)))] = img
    if ind==2:
        wc[int(n/(2**l)):int(n/(2**(l-1))),0:int(m/(2**l))] = img
    if ind==3:
        wc[int(n/(2**l)):int(n/(2**(l-1))),int(m/(2**l)):int(m/(2**(l-1)))] = img
    if ind==0:
       wc[0:int(n/(2**(l))),0:int(m/(2**(l)))] = img
    return wc


def makeonfilter(Type,Par):
    '''    
    
% MakeONFilter -- Generate Orthonormal QMF Filter for Wavelet Transform
%  Usage
%    qmf = MakeONFilter(Type,Par)
%  Inputs
%    Type   string, 'Haar', 'Daubechies'
%    Par    integer, it is a parameter related to the support and vanishing
%           moments of the wavelets, explained below for each wavelet.
%
% Outputs
%    qmf    quadrature mirror filter
%
    '''

    if Type=='Haar':
        f=np.array([1,1],ndmin=2) / np.sqrt(2)
    
    
    if Type == 'Daubechies':
        if Par==4:  
            f = np.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551	],ndmin=2)

        if Par==6: 
            f = np.array([0.332670552950, 0.806891509311, 0.459877502118,	-0.135011020010, -0.085441273882,	0.035226291882	],ndmin=2)
        if Par==8:
            f = np.array([ 	0.230377813309, 0.714846570553, 0.630880767930,	-0.027983769417,	 -0.187034811719,	0.030841381836, 0.032883011667, -0.010597401785],ndmin=2)
    
    
    f = f / np.linalg.norm(f)
    return f



def mconv(f,x):
    """
    
    mconv -- Convolution Tool
    P. Carré XLIM
    Usage
    y = mconv(f,x)
    Inputs
    f   filter
    x   1-d signal
  Outputs
    y   filtered result

  Description
    Filtering by periodic convolution of x with f

    """
    n = x.shape[1]
    p = f.shape[1]
    
    dim=max(n,p)
    
    y=np.real(np.fft.ifft(np.fft.fft(x,dim)*np.fft.fft(f,dim)))
	
    return y 

def banc_filtre2d(sens,s,L,h0):
    '''
    
       res=banc_filtre2d(sens,s,l,h0);
    Décomposition
    	s->image à décomposer
    	l->nbre d'échelle de décomposition (ex l=1 : décomp sur 1 échelle)
    	h0->filtre passe-bas (ex : Daub. 4 h0=[326/675     1095/1309     648/2891    -675/5216] )
    	sens=0 -> décompostion
    	res : la matrice contenant le res de la décomposition (ex si on a une décomp en 1 échelle
     alors res(1:n/2,1:n/2) contient la trame
    
    Reconstruction
    	s->matrice contenant la décomp à reconstruire
    	l->nbre d'échelle 
    	h0->filtre passe-bas 
    	sens=1 -> reconstruction
    res : image reconstruite
     AUTHOR: P. Carré, XLIM Lab. CNRS 7252 University of Poitiers, France
    philippe.carre@univ-poitiers.fr
    
    '''
    
    g0=np.zeros(h0.shape)
    
    for n in range (0,h0.shape[1]):
        g0[0,n]=h0[0,n]*(-1)**(n+1)
    
    g0=g0[0:1,h0.shape[1]::-1]
    
    
    
    if sens==0:
        #res=s.copy()
        
        #Ajout Gestion des tailles non adaptées
        
        n = s.shape[0]
        m = s.shape[1]

                
        res = np.copy(s[0:int(n/(2**L))*(2**L),0:int(m/(2**L))*(2**L)])
        n = res.shape[0]
        m = res.shape[1]
    
        for b in range(0,L):
           # traitement par ligne
           for x in range(0,n):
               ltrame=res[x:x+1,0:m]
               ls2=mconv(h0,ltrame)
               lS2=ls2[0:1,0::2]
               
               ls3=mconv(g0,ltrame)
               lS3=ls3[0:1,0::2]
               
               res[x:x+1,0:m]=np.concatenate((lS2,lS3),axis=1)
               
          #traitement par colonne
           for x in range(0,m):

                ltrame=res[0:n,x:x+1]
                ls2=mconv(h0,ltrame.T)
                lS2=ls2[0:1,0::2]
                ls3=mconv(g0,ltrame.T)
                lS3=ls3[0:1,0::2]
              
                res[0:n,x:x+1]=np.concatenate((lS2,lS3),axis=1).T
               
           n = int(n/2)
           m = int(m/2)
  
    else:
        n = int(s.shape[0]/2**L)
        m = int(s.shape[1]/2**L)
   
        res=np.copy(s)
        for b in range(0,L):
            #Traitement par colonne

            for x in range(0,2*m):
                ltrame=res[0:n,x:x+1].T
                ls=res[n:2*n,x:x+1].T
                
                s2p=np.zeros((1,n*2))
                s2p[0:1,0:n*2:2]=ltrame
                
                S2p=np.fliplr(mconv(h0,np.fliplr(s2p)))
                
                s3p=np.zeros((1,n*2))
                s3p[0:1,0:2*n:2]=ls;
                S3p=np.fliplr(mconv(g0,np.fliplr(s3p)))
                
                    
                
                rec=S2p+S3p
                res[0:2*n,x:x+1]=rec.T
            
            for x in range(0,2*n):
                ltrame=res[x:x+1,0:m]
                ls=res[x:x+1,m:2*m]
                
                s2p=np.zeros((1,m*2))
                s2p[0:1,0:m*2:2]=ltrame
                
                S2p=np.fliplr(mconv(h0,np.fliplr(s2p)))
                
                s3p=np.zeros((1,m*2))
                s3p[0:1,0:2*m:2]=ls
                S3p=np.fliplr(mconv(g0,np.fliplr(s3p)))
                
                rec=S2p+S3p
                res[x:x+1,0:2*m]=rec
 
            n = n*2
            m = m*2

    return res