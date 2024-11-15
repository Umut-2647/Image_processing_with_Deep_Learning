import cv2
import numpy as np


def non_max_supression(boxes,probs=None,overlapThresh=0.3):

    if len(boxes)==0:
        return []
    
    if boxes.dtype.kind=="i": #eğer kutuların tipi integer ise float tipine çevir
        boxes= boxes.astype("float")

    x1= boxes[:,0] #kutuların x1 koordinatları
    y1 = boxes[:,1] #kutuların y1 koordinatları
    x2 = boxes[:,2] #kutuların x2 koordinatları
    y2 = boxes[:,3] #kutuların y2 koordinatları


    #alani bulalim

    area = (x2-x1+1)*(y2-y1+1) #alanı hesapla

    idxs = y2

    #olasilik degerleri

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs) #olasılıkları sırala

    pick=[] #seçilen kutuları tutacak olan liste

    while len(idxs)> 0 :
        last = len(idxs)-1
        i=idxs[last] #olasılığı en yüksek kutuyu seç
        pick.append(i) #seçilen kutuyu ekle

        #en buyuk ve en kucuk x ve y

        xx1 = np.maximum(x1[i],x1[idxs[:last]]) #en büyük x1 koordinatını bul
        yy1 = np.maximum(y1[i],y1[idxs[:last]]) #en büyük y1 koordinatını bul
        xx2 = np.minimum(x2[i],x2[idxs[:last]]) #en küçük x2 koordinatını bul
        yy2 = np.minimum(y2[i],y2[idxs[:last]]) #en küçük y2 koordinatını bul

        #genislik ve yukseklik bul

        w = np.maximum(0,xx2-xx1+1)
        h = np.maximum(0,yy2-yy1+1)

        #overlap 

        overlap= (w*h)/area[idxs[:last]]

        idxs = np.delete(idxs,np.concatenate(([last],np.where(overlap>overlapThresh)[0]))) #overlap değeri belirlediğimiz eşik değerinden büyükse kutuyu sil
    
    return boxes[pick].astype("int") #seçilen kutuları döndür 





