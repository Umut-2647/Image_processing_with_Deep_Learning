import cv2
import os
import matplotlib.pyplot as plt

def img_pyramid(image,scale=1.5,minSize=(224,224)):

    yield image #bu yield ifadesi ile fonksiyonun bir generator olduğunu belirtiyoruz

    while True:
        w= int(image.shape[1]/scale) #resmin yeni boyutlarini belirliyoruz
        image = cv2.resize(image,(w,w))

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]: #eger belirlediğimiz boyutlardan küçükse döngüyü kırıyoruz
            break
    
        yield image


# img =cv2.imread("media/husky.jpg")   #resmi okuyoruz 

# print(img.shape)

# im = img_pyramid(img,scale=1.5,minSize=(10,10)) #resmi fonksiyona gönderiyoruz

# for i ,image in enumerate(im): 
#     print(i)

#     if i==5:
#         plt.imshow(image)