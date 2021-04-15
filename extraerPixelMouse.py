import cv2
import numpy as np

img=cv2.imread('gato3.png')

#x,y son las coordenadas donde se dio click
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
	#indica que el boton izquierdo del mouse fue presionado
    if event == cv2.EVENT_LBUTTONDOWN:
        b,g,r = img[x,y]
        print("{},{},{}".format(b,g,r))

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while(1):
    cv2.imshow("image", img)
    #waitKey regresa 32 bits pero solo importa el primer byte el cual contiene el ascii de la tecla presionada
    #el ascii de la tecla Esc es 27 al presionarla cierra el programa
    if cv2.waitKey(0)&0xFF==27:
        break
cv2.destroyAllWindows()
