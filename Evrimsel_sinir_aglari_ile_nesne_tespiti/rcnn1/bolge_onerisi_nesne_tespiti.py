from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2
import os

from non_max_supression import non_max_supression #ayni obje icin birden fazla kutu cikmasini engellemek icin kullanilir


def selective_search(image):
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() #selective search nesnesi oluştur
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality() 


    print("Selective Search baslatiliyor")
    rects = ss.process()

    return rects[:1000]


#model

model=ResNet50(weights="imagenet") #resnet modelini yükle

path=os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","media","animals.jpg")
image = cv2.imread(path) #resmi oku
image=cv2.resize(image,(400,400))
(H,W)= image.shape[:2]


rects=selective_search(image)

proposals=[]
boxes=[]


for (x,y,w,h) in rects:

    if w/float(W) < 0.1 or h/float(H) < 0.1: 
        continue

    roi=image[y:y+h,x:x+w] #kutuyu al
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB) #renk uzayını değiştir
    roi=cv2.resize(roi,(224,224))

    roi=img_to_array(roi) #diziye çevir
    roi=preprocess_input(roi) #resnete hazırla

    proposals.append(roi) #tahmini ekle
    boxes.append((x,y,w,h))


proposals=np.array(proposals) #listeyi diziye çevir

#predict yani tahmin yap
print("Tahmin yapiliyor(predict kisminda)")
preds= model.predict(proposals) #tahmin yap
preds=imagenet_utils.decode_predictions(preds,top=1) #tahminleri çöz

labels={}
min_conf=0.8

for (i,p) in enumerate(preds):

    (_,label,prob)=p[0] #tahminleri al bize 3 deger dondurur biri id ,digeri label,digeri de olasilik

    if prob>=min_conf:

        (x,y,w,h)=boxes[i] 

        box = (x,y,x+w,y+h) 

        L=labels.get(label,[]) #etiketi al
        L.append((box,prob)) #kutuyu ve olasılığı ekle
        labels[label]=L

clone=image.copy()

for label in labels.keys(): #etiketlerde dolaş
    for (box,prob) in labels[label]: #kutu ve olasılıklarda dolaş
        boxes=np.array([p[0] for p in labels[label]]) 
        proba=np.array([p[1] for p in labels[label]])
        boxes=non_max_supression(boxes,proba)

        for (startX,startY,endX,endY) in boxes: #cizilecek kutuları al  
            cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)
            y=startY-10 if startY-10>10 else startY+10 

            cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

        cv2.imshow("After",clone)

cv2.waitKey(0)
cv2.destroyAllWindows()
