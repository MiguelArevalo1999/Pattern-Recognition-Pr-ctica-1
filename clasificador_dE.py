from math import sqrt
import cv2
import numpy as np

clase1 = []
clase2 = []
clase3 = []

#Se leen los datos del CMA y se insertan en sus clases
f = open('cieloRGB.txt','r')
for i in range(4515):
  mensaje = f.readline()
  clase1.append([int(i) for i in mensaje[:-1].split(' ')])
f.close()

f = open('boscosoRGB.txt','r')
for i in range(35700):
  mensaje = f.readline()
  clase2.append([int(i) for i in mensaje[:-1].split(' ')])
f.close()

f = open('sueloRGB.txt','r')
for i in range(9350):
  mensaje = f.readline()
  clase3.append([int(i) for i in mensaje[:-1].split(' ')])
f.close()


def calcularMedia(clase):
  media = [0,0,0]
  for i in range(len(clase)):
    for j in range(3):
      media[j] += clase[i][j]

  for i in range(3):
    media[i] /= len(clase)

  return media

#Se utiliza el canal rojo y verde
def funcionDiscriminante(claseA,claseB):
  A = [0,0,0]
  for i in range(3):
    if i != 2:
      A[i] = claseA[i] - claseB[i]

  B = [0,0,0]
  for i in range(3):
    if i != 2:
      B[i] = claseA[i] + claseB[i]

  C = [0,0,0]
  for i in range(3):
    if i != 2:
      C[i] = A[i] * B[i]

  D = 0
  for i in range(3):
    if i != 2:
      D += C[i]
  D /= -2

  F = [A[0],A[1],D]

  return F

media1 = calcularMedia(clase1)
media2 = calcularMedia(clase2)
media3 = calcularMedia(clase3)
print(media1)
print(media2)
print(media3)

d12 = funcionDiscriminante(media1,media2)
d13 = funcionDiscriminante(media1,media3)
d23 = funcionDiscriminante(media2,media3)
print(d12)
print(d13)
print(d23)

def probarFD(patron, d):
  res = 0
  for i in range(3):
    if i != 2:
      res += d[i] * patron[i]
  res += d[-1]

  return res

def principal(valores):
  patronDesconocido = valores

  d1 = probarFD(patronDesconocido,d12)
  d2 = probarFD(patronDesconocido,d13)
  d3 = probarFD(patronDesconocido,d23)
  print(d1)
  print(d2)
  print(d3)


  clase = ""
  if d1 >= 0 and d2 >= 0:
    clase = "Cielo"
  elif d1 >= 0 and d3 >= 0:
    clase = "Bosque"
  elif d2 >= 0 and d3 >= 0:
    clase = "Suelo"
  elif d3 >= 0:
    clase = "Bosque"
  elif d1 >= 0:
    clase = "Suelo"

  #Se cambia el nombre de la ventana al de la clase
  cv2.setWindowTitle("image",clase)
  print("Clase: " + clase)

#Se usa para detectar los clic
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
  valores = 0

  if event == cv2.EVENT_LBUTTONDOWN:
    b, g, r = image[y,x]
    valores = [r, g, b]
    print(valores)
    principal(valores)

image = cv2.imread('CMA-x1.png')
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while(1):
    cv2.imshow("image", image)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()