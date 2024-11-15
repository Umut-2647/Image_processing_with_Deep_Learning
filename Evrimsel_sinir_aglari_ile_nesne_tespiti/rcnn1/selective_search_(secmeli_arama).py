import cv2
import random
import os
#bu selective_search fonksiyonu image_pyramid ve sliding_window fonksiyonlarına alternatif olarak kullanılabilir

path = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","media","pyramid.jpg")

image = cv2.imread(path) #resmi oku

image=cv2.resize(image,(600,600)) #resmi boyutlandır


#ilklendir selectivesearch

#selective search nesnesini ice aktar

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() #selective search nesnesi oluştur
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality() 


print("Selective Search baslatiliyor")
rects = ss.process()


output= image.copy()

for (x,y,w,h) in rects[:50]: #50 kutuyu göster

    color = [random.randint(0,255) for j in range(0,3)] #rastgele renk üret

    cv2.rectangle(output,(x,y),(x+w,y+h),color,2) #kutuyu çiz




cv2.imshow("Output",output) #çıktıyı göster
cv2.imshow("Image",image) #resmi göster
cv2.waitKey(0)
cv2.destroyAllWindows()