import cv2
import numpy as np
import os
from yolo_model import YOLO

yolo = YOLO(0.6,0.5)

file = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","yolo_v3","coco_classes.txt")

with open(file) as f: #coco_classes.txt dosyasını okuyoruz
    class_name = f.readlines()

all_classes = [c.strip() for c in class_name]

img_path = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","yolo_v3","dog_cat.jpg")

image =cv2.imread(img_path)

cv2.imshow("Image",image)


pimage = cv2.resize(image,(416,416)) #yolo modeli 416x416 boyutunda resimlerle eğitildiği için resmi bu boyuta çekiyoruz

pimage = np.array(pimage,dtype="float32") #resmi float32 veri tipine çeviriyoruz

pimage/=255.0 #resmi 0-1 arasına çekiyoruz yani normalizasyon yapıyoruz

pimage = np.expand_dims(pimage,axis=0) #resmi 4 boyutlu hale getiriyoruz yolo icin gerekli bir sey

boxes, classes,score =yolo.predict(pimage,image.shape) #resmi yolo modeline veriyoruz ve nesnenin kutu koordinatlarını, sınıfını ve skorunu alıyoruz

print(classes)


for box,score,cl in zip(boxes,score,classes): #her bir nesne için kutu koordinatlarını, skorunu ve sınıfını alıyoruz
    
    x,y,w,h = box #kutu koordinatlarını alıyoruz

    top = max(0,np.floor(x+0.5).astype(int)) #biraz daha iyi gözükmesi için kutuyu biraz büyütüyoruz(pay birakiyoruz)
    left = max(0,np.floor(y+0.5).astype(int))
    right = max(0,np.floor(x+w+0.5).astype(int))
    bottom = max(0,np.floor(y+h+0.5).astype(int))

    cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)  
    cv2.putText(image,"{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)

cv2.imshow("YOLO",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
