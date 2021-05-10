
import cv2
import numpy as np
import matplotlib.pyplot as plt

def largest_contours_rect(saliency):
    #sofi
    #_,contours, hierarchy = cv2.findContours(saliency * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    #tania:
    contours, hierarchy = cv2.findContours(saliency *1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key = cv2.contourArea)
    return cv2.boundingRect(contours[-1])
def openOperation(src):
    dst= src
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7),
                                       (-1, -1))
    dst=cv2.erode(src,element, 3)
    dst=cv2.dilate(src,element, 3)
    
    return dst

def SaliencyDetection(image,numFloodFill=1):
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    image=image.astype('uint8')
    h,w=image.shape[0],image.shape[1]
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    saliencyMap=(255*saliencyMap).astype('uint8')

    #saliencyMap.convertTo(saliencyMap, cv.CV_8U, 255.0);
    
    t,threshMap=cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
    #t,threshMap=cv2.threshold(saliencyMap, 150, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU );
    
    #opcion1:
       
    
    saliencyBinaryMap=openOperation(threshMap)
    
    #plt.imshow(saliencyBinaryMap)
    
    saliencyBinaryMap=(255*saliencyBinaryMap).astype('uint8')
    saliencyBinaryMap[np.where(saliencyBinaryMap > 0)] = cv2.GC_FGD
    
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    cv2.grabCut(image,saliencyBinaryMap,None,bgdModel,fgdModel,1,cv2. GC_INIT_WITH_MASK)
    
    cv2.grabCut(image,saliencyBinaryMap,None,bgdModel,fgdModel,4,cv2.GC_EVAL)
    
    openOperation(saliencyBinaryMap);
     
        
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    for _ in range(numFloodFill):
        cv2.floodFill(saliencyBinaryMap, mask, (0,0), 255);
    
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200),(-1, -1))
    
    #cv2.dilate(saliencyBinaryMap,element, iterDil )
    
    return saliencyBinaryMap
    #opcion 2:
    #kernel=np.ones((5,5),np.uint8)
    # closing = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)
    #saliency=closing*255
    #saliency[np.where(saliency > 0)] = cv2.GC_FGD

    #bgdModel = np.zeros((1,65),np.float64)
    #fgdModel = np.zeros((1,65),np.float64)
    #rect = largest_contours_rect(saliency)

    
    #cv2.grabCut(image,saliency,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)

    #mask2 = np.where((saliency==2)|(saliency==0),0,1).astype('uint8')
    #img = image * mask2[:,:,np.newaxis]

    #return img, mask2

def distanceField(mask):
    
    #Invertir máscara
    mask_inv=mask.copy()
    mask_inv[mask==0 ]=255
    mask_inv[mask==1]=0

   

    #Aplicar el distance Transformation: #nose si normalizo como quiero
    dt,l=cv2.distanceTransformWithLabels(mask_inv, cv2.DIST_L1, 2, labelType=cv2.DIST_LABEL_PIXEL)
    #print(np.unique(dt))


    dt_n=dt/np.max(dt)  #cv2.normalize(dt,0,1,cv2.NORM_INF)

    

    dt_n_filtered = cv2.GaussianBlur(dt_n,(5,5),5) #raro nose si esta bien

    
    
    return dt_n_filtered

def SaliencyDistanceField(image, numFloodFill=1,graph=True):
    #opcion 1
    mask=SaliencyDetection(image,numFloodFill)
    
    #opcion2:
    #_,mask=SaliencyDetection(image)
    
    dt=distanceField(mask)
    
    if graph:
        

        plt.figure(figsize=(20,20))
        plt.subplot(1,2,1)
        plt.title('Saliency Detection')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title('Distance Field')
        plt.imshow(dt, cmap='gray')
        plt.axis('off')
        
        plt.show() 
        
    
    return dt
    
    
