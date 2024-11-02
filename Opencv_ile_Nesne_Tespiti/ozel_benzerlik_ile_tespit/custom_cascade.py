"""
1) veri seti
(n,p)
2) cascade programi indirir
3) cascade
4) cascade kullanarak tespit algoritmasi yaz
"""

import cv2
import numpy as np
import os

#resim depo
path = "images"

#resim boyutu
imgWidht= 180
imgHeight= 120

#video capture

cap = cv2.VideoCapture(0) #0: bilgisayar kamerasi, 1: usb kamera
cap.set(3,640) #3: genislik
cap.set(4,480) #4: yukseklik
cap.set(10,180) #10: parlaklik


global countFolder #resimlerin kaydedilecegi klasor sayisi

def saveDataFunc(): #resimleri kaydetme fonksiyonu
    global countFolder
    countFolder= 0
    while os.path.exists(path+str(countFolder)): #klasor var mi kontrol et
        countFolder+=1 #klasor sayisini arttir
    os.makedirs(path+str(countFolder)) #klasor olustur

saveDataFunc() #resimleri kaydetme fonksiyonunu cagir

count = 0 
countSave= 0

while True:
    success, img = cap.read() #kameradan resim oku

    if success:
        img = cv2.resize(img,((imgWidht,imgHeight))) #resmi yeniden boyutlandir

        if count%5==0: #her 5 karede bir resimleri kaydet
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img) #resmi kaydet
            countSave+=1 #kaydedilen resim sayisini arttir
            print(countSave) #kaydedilen resim sayisini yazdir
        count+=1

        cv2.imshow("Video",img)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()










