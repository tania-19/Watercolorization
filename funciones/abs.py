
import numpy as np
import matplotlib.pyplot as plt

import cv2

def clamp(n,a,b):
    if (n<a):
        out = a
    elif (n>b):
        out = b
    else:
        out = n
        
    if out % 2 == 0:#impar
        out+=1
        
    return out

def Abstraction(img, saliency, d,graph=False, thresh_1=0.05, minSize_1=2,thresh_2=0.1, minSize_2=2,spatialW=0.01, slicPixelSize=16, numslicIter=4  ):
    

    HSF = cv2.hfs.HfsSegment_create(img.shape[0], img.shape[1],thresh_1, minSize_1,thresh_2, minSize_2,spatialW, slicPixelSize, numslicIter )

    
    #regiones:
    tag = HSF.performSegmentGpu(img,False)
    #regiones pintadas con average
    img_ave= HSF.performSegmentGpu(img,True) 
    
    
    #Filter:
    M,N=tag.shape
    J=img_ave.copy()
        

    ancho=5

    

    regiones=np.unique(tag)




    regiones_saliency=np.unique(tag[saliency==1])

    #regiones dentro del saliency:
   
    for  r in regiones_saliency  :
        mask= tag == r  #un canal
        #si estoy en region saliency:

        blur=J
        blur=cv2.GaussianBlur(blur,(ancho,ancho), 0, 0)

        J[:,:,0] = J[:,:,0] + ( ( blur[:,:,0] - J[:,:,0]) * mask )
        J[:,:,1] = J[:,:,1] + ( ( blur[:,:,1] - J[:,:,1]) * mask )
        J[:,:,2] = J[:,:,2] + ( ( blur[:,:,2] - J[:,:,2]) * mask )

    
    #regiones fuera del saliency:
    regiones_pasadas=[]

    for i in range(M):
        for j in range(N):

            region=tag[i,j]



            if region not in regiones_pasadas:
                regiones_pasadas.append(region)
                if region not in regiones_saliency: 


                    dd=d[i,j]



                    K=clamp( int (10 * (dd + 0.3) ) , 4, 9)

                    mask=  d < 1.3*dd
                    mask2= saliency == 0 #no soy saliency


                    regiones_extras=np.unique(tag[mask*mask2])
                    for rx in regiones_extras:
                        if rx not in regiones_pasadas:
                            regiones_pasadas.append(rx)




                    blur=J
                    blur=cv2.GaussianBlur(blur,(4+K,4+K), 0, 0)

                    J[:,:,0] = J[:,:,0] + ( ( blur[:,:,0] - J[:,:,0]) * mask * mask2 )
                    J[:,:,1] = J[:,:,1] + ( ( blur[:,:,1] - J[:,:,1]) * mask * mask2 )
                    J[:,:,2] = J[:,:,2] + ( ( blur[:,:,2] - J[:,:,2]) * mask * mask2 )


        

    
    
    if graph:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img_ave_RGB = cv2.cvtColor(img_ave, cv2.COLOR_LAB2RGB)
        J_RGB = cv2.cvtColor(J, cv2.COLOR_LAB2RGB)
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,2,1)
        
        plt.axis('off')
        plt.imshow(img_RGB)
        plt.title('Imagen')
        plt.subplot(2,2,2)
        plt.imshow(tag)
        plt.title('Regiones')
        plt.axis('off')
        
        plt.subplot(2,2,3)
        plt.imshow(img_ave_RGB)
        plt.title('HSF')
        plt.axis('off')
        
        plt.subplot(2,2,4)
        plt.title('Salida del Abstration')
        plt.imshow(J_RGB)
        plt.axis('off')
        plt.show()
    
    return J


    
    
