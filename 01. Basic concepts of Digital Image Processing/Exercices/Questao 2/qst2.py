# -*- coding: cp1252 -*-
import numpy as np
import cv2

#Leitura da Imagem e transforma��o em uma de tons cinzas
img = cv2.imread('perucaba.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplica��o de Suaviza��o
suave = cv2.GaussianBlur(img, (7, 7), 0)

#Aplica��o de Segmenta��o
canny1 = cv2.Canny(suave, 1, 15)
canny2 = cv2.Canny(suave, 1, 15)

resultado = np.vstack([
np.hstack([img, suave ]),
np.hstack([canny1, canny2])
])

imagem, PerocabaPic = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("1. Segmenta��o e Detecta��o de Bordas com o Canny.jpg", resultado)
cv2.imwrite("2. NewPerocaba.jpg", PerocabaPic)
cv2.waitKey(0)
