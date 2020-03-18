# -*- coding: cp1252 -*-
import numpy as np
import cv2
import mahotas
from matplotlib import pyplot as plt

#Função para escrita na imagem
def escreve(img, texto, cor = (255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,
     cv2.LINE_AA)

imgColorida = cv2.imread('moedas.jpg') #Carregamento da imagem

#Conversão para tons de cinza
img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Cinza.jpg", img)

#Blur/Suavização da imagem
suave = cv2.blur(img, (7, 7))

#Binarização que resulta em pixels brancos e pretos
T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)
cv2.imwrite("Imagem Binarizada.jpg", img)

#Detecção de bordas com Canny
bordas = cv2.Canny(bin, 70, 150)

#Identificação e contagem dos contornos da imagem
(lx, objetos, lx) = cv2.findContours(bordas.copy(),
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

temp = np.vstack([
    np.hstack([img, suave]),
    np.hstack([bin, bordas])
    ])

imgC2 = imgColorida.copy()
cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
escreve(imgC2, str(len(objetos))+" moedas foram encontradas!")
cv2.imwrite("Resultado.jpg", imgC2)
#cv2.imwrite("Cinza.png", bin)

n, bins, patches = plt.hist(bin.ravel(), 256, [1, 255])
plt.show()

#for y in range(0, len(objetos)):
#    n, bins, patches = plt.hist(objetos[y].ravel(), 256, [1, 255])
#    plt.show()
    #print(y)

cv2.waitKey(0)
