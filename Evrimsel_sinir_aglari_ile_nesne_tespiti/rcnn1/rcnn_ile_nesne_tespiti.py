import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array
import os

path = os.path.join(os.getcwd(), "Image_processing_with_Deep_Learning", "Evrimsel_sinir_aglari_ile_nesne_tespiti", "media", "mnist.png")
image = cv2.imread(path)

cv2.imshow("Image", image)

#selective search nesnesini ice aktar

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() #selective search nesnesi oluştur
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality() 


print("Selective Search baslatiliyor")
rects = ss.process()

proposals=[] #tahminlerin tutulacagi liste

boxes=[] #kutuların tutulacagi liste

output = image.copy()

for (x,y,w,h) in rects[:500]: #20 kutuyu göster
    color = [random.randint(0,255) for j in range(0,3)] #rastgele renk üret

    cv2.rectangle(output,(x,y),(x+w,y+h),color,2)

    roi = image[y:y+h,x:x+w] #ilk yukseklik sonra genislik

    #cv2.INTER_LANCZOS4
    #Görüntüyü büyütme ve küçültme işlemlerinde detayların korunmasını ve daha az bulanıklık elde edilmesini sağlar.

    roi = cv2.resize(roi,dsize=(32,32),interpolation=cv2.INTER_LANCZOS4) #resmi boyutlandır

    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #renk uzayını değiştir

    roi=img_to_array(roi) #diziye çevir

    proposals.append(roi) #tahmini ekle

    boxes.append((x,y,x+w,y+h)) 

proposals = np.array(proposals,dtype="float64") 
boxes = np.array(boxes,dtype="int32")

#modeli yükle
model_path = os.path.join(os.getcwd(),"rakam_siniflandirma_agirlik","model_trained_new.p")

print("Siniflandirma")
pickle_in = open(model_path,"rb") 

model = pickle.load(pickle_in) #modeli yükle

proba = model.predict(proposals) #tahmin yap

number_list = []
idx = []

for i in range (len(proba)): #tahminler arasında dolaş
    max_prob = np.max(proba[i]) #en yüksek olasılığı al
    if max_prob>0.95: #eğer olasılık %95 ten büyükse
        idx.append(i) #indexi ekle
        number_list.append(np.argmax(proba[i]))  #en yüksek olasılığın indexini ekle


for i in range(len(number_list)): #olasiliklar arasında dolaş

    j = idx[i] #indexi al       (x,y)                 (x+w,y+h)          
    cv2.rectangle(image,(boxes[j][0],boxes[j][1]),(boxes[j][2],boxes[j][3]),(0,255,0),2)

    image = cv2.putText(image,str(np.argmax(proba[j])),(boxes[j][0]+5,boxes[j][1]+5),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),1)

    cv2.imshow("Output",image) #çıktıyı göster

cv2.waitKey(0)
cv2.destroyAllWindows()

