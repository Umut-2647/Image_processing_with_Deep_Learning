import cv2
import numpy as np
from collections import deque

#nesne merkezini depolayacak veri tipi
buffer_size = 16
pts = deque(maxlen=buffer_size)

#mavi renk aralığı HSV fomratında

bluelower = (84,98,0)
blueupper = (179,255,255)

#capture

cap = cv2.VideoCapture(0) #bilgisayarın webcam 
cap.set(3,960) #genişlik
cap.set(4,480) #yükseklik

while True:
    success , imgOriginal = cap.read() #kameradan görüntü al

    if success:
        #blur islemi yapiyoruz
        blurred = cv2.GaussianBlur(imgOriginal,(11,11),0)

        #HSV formatına dönüştürüyoruz
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",hsv)

        #mavi için maske oluşturuyoruz
        mask = cv2.inRange(hsv,bluelower,blueupper)
    
        cv2.imshow("mask Image",mask)

        #maskeniin etrafinda kalan gurultuleri temizliyoruz

        mask= cv2.erode(mask,None,iterations=3)
        mask = cv2.dilate(mask,None,iterations=3)   
        cv2.imshow("mask+erozyon genisleme Image",mask)
        
        #kontur tespiti
        contours,_=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        center = None

        if len(contours)>0:
            #en buyuk konturu aliyoruz
            c=max(contours,key=cv2.contourArea)

            #dikdortgene ceviriyoruz
            rect = cv2.minAreaRect(c) #minimum alana sahip dikdörtgen dondurur
            
            ((x,y),(width,height),rotation) = rect #dikdörtgenin merkezini, genişliğini, yüksekliğini ve döndürülme açısını alır

            s = "x: {} , y: {} ,width: {}, height : {}, rotation :{}".format(np.round(x),np.round(y),np.around(width),np.around(height),np.around(rotation))
                       #yaklaşık değerler alır
            print(s)

            #kutucuk
            box = cv2.boxPoints(rect) #dikdörtgenin köşelerini alır
            box = np.int64(box)      

            #moment 
            M = cv2.moments(c) #merkez noktasını alır
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])) #merkez noktasını alır

            #konturu çizdiriyoruz sari

            cv2.drawContours(imgOriginal,[box],0,(0,255,255),2)

            #merkeze bir nokta cizdirelim

            cv2.circle(imgOriginal,center,5,(255,0,255),-1)

            #bilgileri ekrana yazdir

            cv2.putText(imgOriginal,s,(25,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)

        #deque yapısına merkez noktasını ekleyelim
        # eğer nesne kaybolursa merkez noktası da kaybolacak
        # bu sayede nesnenin yolu çizilecek
        

        pts.appendleft(center)

        for i in range(1,len(pts)):

            if pts[i-1] is None or pts[i] is None: continue #eğer merkez noktası yoksa döngüyü atla

            cv2.line(imgOriginal,pts[i-1],pts[i],(0,255,0),3) #çizgi çizdir




        cv2.imshow("Original Image",imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()