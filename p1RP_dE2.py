from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import io
import numpy as np
import cv2
from skimage import io

image = cv2.imread('CMA-x1.png')

#Convierte la imagen a escala de grises
gray = rgb2gray(image)
h, w = gray.shape

#Binariza la imagen
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0

#Filtro maximo
def maxBoxFilter(n, path_to_image):
    img = cv2.imread(path_to_image)

    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)
    cv2.imwrite('BN.jpg', imgResult)
    # Shows the result
    # Adjust the window length
    cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL)
    cv2.imshow('Result with n ' + str(n), imgResult)

#Filtro minimo
def minBoxFilter(n, path_to_image):
    img = cv2.imread(path_to_image)

    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.erode(img, kernel)
    cv2.imwrite('BN.jpg', imgResult)
    # Shows the result
    cv2.namedWindow('Result min with n ' + str(n),cv2.WINDOW_NORMAL)  # Adjust the window length
    cv2.imshow('Result min with n ' + str(n), imgResult)

#Se guarda la imagen binarizada
filename = 'BN.jpg'
#Como solo tiene 0 y 1 la imagen quedaria en negro
#Se multiplica por 255 para que el 1 quede en 255 
cv2.imwrite(filename, 255*gray)
#Se aplica filtro minimo y maximo
maxBoxFilter(3, filename)
minBoxFilter(5, filename)

filename2 = 'boscoso.jpg'
filename3 = 'cieloSuelo.jpg'

image2 = cv2.imread("BN.jpg")
#Se aplica el OR para sacar la zona boscosa en RGB
OR = cv2.bitwise_or(image, image2)
cv2.imshow('OR', OR)
cv2.imwrite(filename2, OR)
#Se aplica el NOT para obtener las clases restantes
mask_inv = cv2.bitwise_not(image2)
cv2.imshow('NOT', mask_inv)
#Se aplica el OR para tenerlas en RGB
OR2 = cv2.bitwise_or(image, mask_inv)
cv2.imshow('OR2', OR2)
cv2.imwrite(filename3, OR2)

#Proceso para meter la zona boscosa al CMA
imageOR = cv2.imread('boscoso.jpg')
lw = [255, 255, 255]

f1 = open("boscosoRGB.txt", "w+")
for i in range(h):
    for j in range(w):
        #Evitamos el color blanco
        if list(imageOR[i, j]) != lw:
            b = imageOR[i, j][0]
            g = imageOR[i, j][1]
            r = imageOR[i, j][2]
            #Evitamos que un valor casi blanco se meta a la clase
            if r < 240 and g < 240 and b < 240:
                f1.write(str(r)+' ')
                f1.write(str(g)+' ')
                f1.write(str(b)+'\n')
f1.close()

#Proceso para meter la zona cielo y suelo al CMA
imageOR2 = cv2.imread('cieloSuelo.jpg')
f2 = open("cieloRGB.txt", "w+")
f3 = open("sueloRGB.txt", "w+")

for i in range(h):
    for j in range(w):
        #Evitamos el color blanco
        if list(imageOR2[i, j]) != lw:
            b = imageOR2[i, j][0]
            g = imageOR2[i, j][1]
            r = imageOR2[i, j][2]
            #En caso de que el valor azul sea mayor que el rojo
            #Se mete a la clase cielo en caso contrario a suelo
            if b > r:
                f2.write(str(r)+' ')
                f2.write(str(g)+' ')
                f2.write(str(b)+'\n')
            elif r < 240 and g < 240 and b < 240:
                f3.write(str(r)+' ')
                f3.write(str(g)+' ')
                f3.write(str(b)+'\n')
f2.close()
f3.close()

while(1):
    cv2.imshow("image", image2)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()