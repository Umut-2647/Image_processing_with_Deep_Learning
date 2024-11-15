from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2
import os

from slinding_window import slinding_window
from image_pyramid import img_pyramid
from non_max_supression import non_max_supression



#ilklendirme parametreleri

WIDTH=600
HEIGHT=600
PYR_SCALE=1.5
WIN_STEP=16
ROI_SIZE=(200,150)
INPUT_SIZE=(224,224) #veriseti boyutlarına uygun olarak ayarlandı

print("Resnet yukleniyor")

model=ResNet50(weights ="imagenet",include_top=True) #resnet modelini yükle

path = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","media","husky.jpg")
orig = cv2.imread(path) #resmi oku

orig= cv2.resize(orig,(WIDTH,HEIGHT)) #resmi boyutlandır
#cv2.imshow("Original",orig) #resmi göster

(W,H)= orig.shape[:2] #resmin boyutlarını al

#image pyramid

pyramid = img_pyramid(orig,scale=PYR_SCALE,minSize=ROI_SIZE) #resmi piramit haline getir

rois=[] 
locs=[]

for image in pyramid: #piramit haline getirilmiş resimlerde dolaş

    scale = W/float(image.shape[1]) #ölçeklendirme oranını hesapla
    for (x,y,roiOrig) in slinding_window(image,step=WIN_STEP,ws=ROI_SIZE):  #pencereyi kaydır

        x=int(x*scale) #koordinatlari ölçeklendir
        y=int(y*scale)
        w=int(ROI_SIZE[0]*scale)
        h=int(ROI_SIZE[1]*scale)

        roi=cv2.resize(roiOrig,INPUT_SIZE) #pencereyi boyutlandır

        roi=img_to_array(roi) #pencereyi diziye çevir

        roi=preprocess_input(roi) #resnete hazırla

        rois.append(roi) #pencereyi listeye ekle
        locs.append((x,y,x+w,y+h))


rois=np.array(rois,dtype="float32") #pencere listesini diziye çevir

print("Siniflandirma yapiliyor")


preds=model.predict(rois) #pencere listesini tahmin et

preds=imagenet_utils.decode_predictions(preds,top=1) #tahminleri çözümle

print(preds)

labels={}
min_conf = 0.9 #en az %90 olasılık

for (i,p) in enumerate(preds):

    (_,label,prob)=p[0] #bize 3 deger dondurur biri id ,digeri label,digeri de olasilik
    
    if prob>=min_conf: #olasılık belirlediğimiz olasılıktan büyükse
        box = locs[i] #kutuyu al

        L=labels.get(label,[]) #etiketi al
        L.append((box,prob))

        labels[label]=L


for label in labels.keys():

    clone = orig.copy() #resmi kopyala(orijinal goruntunun uzerine isaretleme yapmak icin)

    for (box,prob) in labels[label]:

        (startX,startY,endX,endY)=box #kutunun koordinatlarını al

        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)

    cv2.imshow("Ilk",clone)

    clone= orig.copy()

    #non maxi supression yapip kutulari birlestir
    #ayirmamiz gerekir cunku non-max-supression fonksiyonunda (boxes,probs) şeklinde iki parametre alır
    boxes=np.array([p[0] for p in labels[label]]) #kutuları al
    probs=np.array([p[1] for p in labels[label]]) 

    boxes=non_max_supression(boxes,probs) #non-max-supression yap

    for (startX,startY,endX,endY) in boxes: 
        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)

        y=startY-10 if startY-10>10 else startY+10 

        cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

    cv2.imshow("Maxima",clone)


cv2.waitKey(0)
cv2.destroyAllWindows()


