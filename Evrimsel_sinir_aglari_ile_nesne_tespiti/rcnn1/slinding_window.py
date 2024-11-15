import cv2
import matplotlib.pyplot as plt
import os


def slinding_window(image,step,ws): #step: kaydırma miktarı, ws: pencere boyutu

    for y in range(0,image.shape[0]-ws[1],step): #y ekseninde kaydırma

        for x in range(0,image.shape[1]-ws[0],step): #x ekseninde kaydırma

            yield(x,y,image[y:y+ws[1],x:x+ws[0]]) #pencereyi döndür


# path = os.path.join(os.getcwd(),"Image_processing_with_Deep_Learning","Evrimsel_sinir_aglari_ile_nesne_tespiti","media","husky.jpg")

# img = cv2.imread(path) #resmi oku

# im= slinding_window(img,5,(200,150)) #pencereyi kaydır

# for i , image in enumerate(im): 
#     print(i)

#     if i==120: #120. pencereyi göster

#         print(image[0],image[1]) #pencerenin koordinatlarını yazdır
#         plt.imshow(image[2]) #pencereyi göster
#         plt.show() 



