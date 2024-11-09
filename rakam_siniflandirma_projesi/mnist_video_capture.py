import cv2
import pickle
import numpy as np




def preProcess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #resmi griye çevirir
    img = cv2.equalizeHist(img) #resmi 255 e kadar genislettik
    img = img/255 #resmi normalize eder

    return img





cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) #kamerayı açar 0 pc kamerası 1 usb kamera 
cap.set(3,480)
cap.set(4,480)

pickle_in = open(r"C:\Users\umuty\Desktop\Image_Processing_with_Deep_Learning\Image_processing_with_Deep_Learning\rakam_siniflandirma_projesi\model_trained_new.p","rb") #modeli yükle
model = pickle.load(pickle_in)

while True:
    success,frame = cap.read()


    img = np.asarray(frame)
    img = cv2.resize(img,(32,32))
    img= preProcess(img)

    img = img.reshape(1,32,32,1)

    #tahmin 
    classIndex = int(np.argmax(model.predict(img))) #tahmin edilen sınıfı alır 

    predictions = model.predict(img) #tahmin edilen sınıfın olasılığını alır

    probVal = np.amax(predictions) #olasılık değerini alır

    print(classIndex,probVal) 

    if probVal>0.7: #eğer olasılık değeri 0.7 den büyükse tahmin edilen sınıfı ve olasılık değerini ekrana yazdırır
        cv2.putText(frame,str(classIndex)+"   "+str(probVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)
    
    cv2.imshow("Rakam siniflandirma",frame)

    if cv2.waitKey(1) & 0xFF== ord("q"): break

cap.release()
cv2.destroyAllWindows()




