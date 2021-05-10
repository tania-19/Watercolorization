#Librerias a utilizar:
import os
import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt


def image_stats(image):
    '''Esta función de el vector de características de la imagen,
    es decir vuelve la media y la desviación estandar de cada canal'''
    
    (l, a, b) = cv2.split(image)
    (l_mean, l_std) = (l.mean(), l.std())
    (a_mean, a_std) = (a.mean(), a.std())
    (b_mean, b_std) = (b.mean(), b.std())

    
    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)


def extraigo_features_dataset():
    grupos=[]
    num_clusters=14
    for i in range(num_clusters):
        path='./styleLib/'+ str(i)
        dirs=os.listdir(path)
        g=[]
        for img in dirs:
            pic=cv2.imread(os.path.join(path,img))

            #Paso todas las imágenes a LAB
            pic=cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)
            g.append(pic)
        grupos.append(g)

    #Extraigo el vector de características promedio en cada grupo:
    features=[]
    for k in range(num_clusters): 
        features_group=[]
        for img in grupos[k]:
            f=image_stats(img) #(l_mean, l_std, a_mean, a_std, b_mean, b_std)
            features_group.append(f)
        features_group=np.array(features_group)

        L_m  =   np.mean(   features_group[:,0]   )
        L_std=   np.mean(   features_group[:,1]   )
        a_m  =   np.mean(   features_group[:,2]   )
        a_std=   np.mean(   features_group[:,3]   )
        b_m  =   np.mean(   features_group[:,4]   )
        b_std=   np.mean(   features_group[:,5]   )

        features.append([L_m ,L_std,a_m,a_std,b_m,b_std])
    return features,grupos

def color_transfer(stat_source, img):
    
    '''Realiza la transferencia de color, 
    stat_source son las vector de característica medio
    img: la imagen que queremos varia el color
    
    '''
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")

    # Describo características deseadas y las que tiene la imagen
    (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = stat_source
    (l_mean_dest, l_std_dest, a_mean_dest, a_std_dest, b_mean_dest, b_std_dest) = image_stats(img)

    # Resto el valor medio
    (l, a, b) = cv2.split(img)
    l -= l_mean_dest
    a -= a_mean_dest
    b -= b_mean_dest

    # Escalado
    l = (l_std_dest / l_std_src) * l
    a = (a_std_dest / a_std_src) * a
    b = (b_std_dest / b_std_src) * b

    # Agrego valor medio de source
    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    # Mantengo numeros entre 0 y 255
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Combino canales
    transfer = cv2.merge([l, a, b])
    #transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    #Imagen modificada en color
    return transfer.astype("uint8")


def show_image(title, image, width = 500):
    
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    # Show the resized image
    plt.title(title)
    plt.imshow(resized)
    plt.axis('off')
    plt.show()
    
    
def Color_adjustment(img):
    #extraigo features del dataset:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    
    features_dataset, dataset= extraigo_features_dataset() #todo en espacio color LAB

    features_img=image_stats(img) #features de la imagen

    #me fijo la feature mas cercana:
    features_dataset=np.array(features_dataset)

    features_img=np.array(features_img)
    dist= np.sqrt(np.sum( (features_dataset- features_img )**2, axis=1))
    ind=np.argmin(dist)
    transferred = color_transfer(features_dataset[ind] , img) 
    return transferred
    
    
   
    
    
