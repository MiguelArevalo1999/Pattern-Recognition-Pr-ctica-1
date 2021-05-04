from math import sqrt
import cv2
import numpy as np

clase1 = [[200,160,120],[210,170,130],[215,172,133],[210,165,134],[198,177,138],
         [146,112,75],[200,168,130],[222,188,150],[148,113,81],[199,164,126],
         [214,179,147],[171,137,103],[192,158,121],[211,177,142],[188,139,103],
         [170,126,88],[157,123,86],[211,172,131],[173,138,106],[188,156,116],]
          
clase2 = [[90,130,60],[92,138,54],[87,128,66],[91,134,60],[85,123,55],
          [93,56,22],[78,47,18],[125,105,55],[123,154,38],[99,119,21],
          [82,97,0],[97,69,43],[131,151,43],[73,63,14],[180,204,41],
          [128,95,45],[89,67,39],[90,119,16],[85,71,29],[136,160,23],]

clase3 = [[30,44,178],[20,40,180],[24,42,184],[28,50,176],[22,46,181],
          [224,227,234],[215,224,229],[181,189,213],[223,228,233],[177,199,209],
          [217,221,229],[185,200,213],[226,229,234],[228,231,236],[208,217,226],
          [199,218,224],[137,168,188],[205,213,224],[216,226,228],[147,175,196],]

click = 0

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
  global click
  global clase1, clase2, clase3
  valores = 0

  if event == cv2.EVENT_LBUTTONDOWN:
    b, g, r = image[y, x]
    valores = [r, g, b]
    if click == 0:
      print("Pixeles Cielo")
    elif click == 19:
      print("Pixeles Bosque")
    elif click == 39:
      print("Pixeles Suelo")
    elif click == 59:
      print("YAAAAAA")

    if click < 20:
        clase1.append(valores)
        #print(clase1)
    elif click < 40:
        clase2.append(valores)
        #print(clase2)
    else:
        clase3.append(valores)
        #print(clase3)

    click += 1


image = cv2.imread('CMA-x1.png')
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while(1):
    print("Comenzar dando click en pixel del cielo 20 veces, luego bosque 20 veces y por ultimo suelo 20 veces")
    cv2.imshow("image", image)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()

def calcularMedia(clase1):
  media = [0,0,0]
  for i in range(len(clase1)):
    for j in range(3):
      media[j] += clase1[i][j]

  for i in range(3):
    media[i] /= len(clase1)

  return media

#Se utiliza el canal rogo y verde
def funcionDiscriminante(claseA,claseB):
  A = [0,0,0]
  for i in range(2):
    A[i] = claseA[i] - claseB[i]

  B = [0,0,0]
  for i in range(2):
    B[i] = claseA[i] + claseB[i]

  C = [0,0,0]
  for i in range(2):
    C[i] = A[i] * B[i]

  D = 0
  for i in range(2):
    D += C[i]
  D /= -2

  F = [A[0],A[1],D]

  return F

def probarFD(patron, d):
  res = 0
  for i in range(2):
    res += d[i] * patron[i]
  res += d[-1]

  return res

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

while(1):
  patronDesconocido = [int(item) for item in input().split()]

  d1 = probarFD(patronDesconocido,d12)
  d2 = probarFD(patronDesconocido,d13)
  d3 = probarFD(patronDesconocido,d23)
  print(d1)
  print(d2)
  print(d3)

  #clase 1: suelo
  #clase 2: bosque
  #clase 3: cielo

  clase = 0
  if d1 >= 0 and d2 >= 0:
    clase = 1
  elif d1 >= 0 and d3 >= 0:
    clase = 2
  elif d2 >= 0 and d3 >= 0:
    clase = 3

  print("Clase: " + str(clase))