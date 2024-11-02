import cv2



object_name = "kalem"

frame_width =280
frame_height = 360

color = (255,0,255)

cap =cv2.VideoCapture(0)
cap.set(3,frame_width) #3: width
cap.set(4,frame_height) #4: height

#trackbar 
cv2.namedWindow("Sonuc") #pencere adı
cv2.resizeWindow("Sonuc",frame_width,frame_height+100) 


def empty(x):
    pass

cv2.createTrackbar("Scale","Sonuc",400,1000,empty) #400: default value, 1000: max value
cv2.createTrackbar("Neighbor","Sonuc",4,50,empty)   

cascade = cv2.CascadeClassifier(r"C:\Users\umuty\Desktop\Image_Processing_with_Deep_Learning\Image_processing_with_Deep_Learning\Opencv_ile_Nesne_Tespiti\ozel_benzerlik_ile_tespit\cascade.xml")

while True:
    success , img = cap.read()

    if success:

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #detection parameters
        #eger scale 0 ise hata verir
        scaleVal = 1+(cv2.getTrackbarPos("Scale","Sonuc")/1000) #0.4

        neighbor =1+(cv2.getTrackbarPos("Neighbor","Sonuc")) 

        #detect
        rects = cascade.detectMultiScale(gray,scaleVal,neighbor) 

        for (x,y,w,h) in rects:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,object_name,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)

        
        cv2.imshow("Sonuc",img)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

