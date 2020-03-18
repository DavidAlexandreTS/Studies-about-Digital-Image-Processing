# -*- coding: cp1252 -*-
import numpy as np
import cv2

#Leitura da Imagem e transformação em uma de tons cinzas
img = cv2.imread('perucaba.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplicação de Suavização
suave = cv2.GaussianBlur(img, (7, 7), 0)

#Aplicação de Segmentação
canny1 = cv2.Canny(suave, 1, 15)
canny2 = cv2.Canny(suave, 1, 15)

resultado = np.vstack([
np.hstack([img, suave ]),
np.hstack([canny1, canny2])
])

imagem, PerocabaPic = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("1. Segmentação e Detectação de Bordas com o Canny.jpg", resultado)
cv2.imwrite("2. NewPerocaba.jpg", PerocabaPic)
cv2.waitKey(0)
