import numpy as np
import os
import cv2

CLASSES=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
         "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
         "sheep","sofa","train","tvmonitor"] #tespit edilebilecek nesneler

COLORS = np.random.uniform(0,255,size=(len(CLASSES),3)) #nesnelerin etiketlerine rastgele renkler atama

txt_path = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","ssd"
                        ,"MobileNetSSD_deploy.prototxt.txt") #txt dosyasinin yolunu belirleme


model_path = os.path.join(os.getcwd(),"MobileNetSSD_deploy.caffemodel") #modelin yolunu belirleme

net = cv2.dnn.readNetFromCaffe(txt_path,model_path) #txt dosyasi ve modeli okuma


dizin = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti",
                     "ssd","media") #resimleri almak istedğimiz dizini belirleme


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #kamerayi acma


while True: #her resim icin dongu

    success,image = cap.read() #kameradan resim okuma
    
    (h,w)= image.shape[:2] #yukseklik ve genislik alma

    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,(300,300),127.5) #ssd modeli icin gerekli olan parametreler

    net.setInput(blob) #blob verisini modele girdi olarak verme

    detections = net.forward() #nesne tespiti yapma

    for j in np.arange(0,detections.shape[2]): #tespit edilen nesneleri gorsellestirme

        confidence = detections[0,0,j,2] #nesne tespiti icin guven degeri

        if confidence> 0.10:
            
            idx = int(detections[0,0,j,1]) #tespit edilen nesnenin indeksi

            box = detections[0,0,j,3:7]*np.array([w,h,w,h]) #tespit edilen nesnenin kutusunu alma

            (startX,startY,endX,endY) = box.astype("int") #kutunun baslangic ve bitis koordinatlari

            label="{} : {}".format(CLASSES[idx],confidence) #tespit edilen nesnenin etiketi ve guven degeri

            cv2.rectangle(image,(startX,startY),(endX,endY),COLORS[idx],2) #tespit edilen nesnenin etiketini ve kutusunu cizme

            y = startY-16 if startY-16>15 else startY+16 #yazinin yazilacagi y koordinati
            cv2.putText(image,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2) #tespit edilen nesnenin etiketini yazdirma

    cv2.imshow("Output",image)

    if cv2.waitKey(1) & 0xFF== ord("q"): break


cv2.destroyAllWindows()        

