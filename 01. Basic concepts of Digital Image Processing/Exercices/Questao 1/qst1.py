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
cv2.imwrite("01. Imagem Cinza.jpg", img)

#Blur/Suavização da imagem
suave = cv2.blur(img, (7, 7))

#Binarização que resulta em pixels brancos e pretos
T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)
cv2.imwrite("02. Imagem Binarizada.jpg", bin)

#Detecção de bordas com Canny
bordas = cv2.Canny(bin, 70, 150)

#Identificação e contagem dos contornos da imagem
(lx, objetos, lx) = cv2.findContours(bordas.copy(),
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

temp = np.vstack([
    np.hstack([img, suave]),
    np.hstack([bin, bordas])
    ])

#Desenho dos contornos
imgC2 = imgColorida.copy()
final = cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
escreve(imgC2, str(len(objetos))+" moedas foram encontradas!")
cv2.imwrite("03. Imagem com as Bordas Destacadas.jpg", imgC2)

n_moedas = 0;
for i in objetos:
    n_moedas += 1
    M = cv2.moments(i)

    #Cálculo da Área
    area = cv2.contourArea(i)

    #Cálculo da posição a partir do Centroide
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    posicao = (x - 150, y)

    #Cálculo do Perímetro
    perimetro = cv2.arcLength(i, True)

    print("Objeto {}:\nArea: {}\nCentro: ({},{})\nPerimetro: {}\n".format(n_moedas, area, x, y, perimetro))

cv2.waitKey(0)
