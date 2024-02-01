import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

grosordebrocha = 25 #en pixeles
grosordeborrador = 100 #en pixeles
tiempoentrecapturas = 5  #Tiempo en segundos entre capturas
tiempoultimacaptura = time.time() - tiempoentrecapturas  #inicializar el tiempo de la última captura

#Interfaz de usuario
folderPath = "Interfaz" ##carpeta de la interfaz
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]##Poner las imagenes de la carpeta como header

##Configuraciones de la captura con cv2
cap = cv2.VideoCapture(0) #Seleccion de camara
#Resolucion de la pantalla
cap.set(3, 1280) 
cap.set(4, 720)

##Inicializacion de componentes(tracking, canvas,brocha)
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
imgCanvas = np.zeros((720, 1280, 3), np.uint8) ##Inicializando canvas en negro
drawColor = (255, 255, 255)   ##Incializando color de la brocha
xp, yp = 0, 0   ##Inicializando posicion de la brocha

##Bucle del Programa
while True:
    #Cargar Video
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Detectar Manos
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    ##Usando el Modulo HandTracking para detectar los dedos
    if lmList is not None and len(lmList) != 0:
        # Indice y medio levantados
        x1, y1 = lmList[8][1:] ## Dedo indice
        x2, y2 = lmList[12][1:] ##Dedo Medio
        fingers = detector.fingersUp() ##Detectar si los dedos estan levantados
        # Si dedo indice y medio están levantados
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Modo seleccion")
            # posición de los dedos y acciones de seleccion de color
            if y1 < 125:
                if 41 < x1 < 141:
                    drawColor = (80, 72, 252) 
                elif 201 < x1 < 301:
                    drawColor = (34, 200, 248)
                elif 361 < x1 < 461:
                    drawColor = (66, 231, 181)
                elif 521 < x1 < 621:
                    drawColor = (255, 173, 89)
                elif 681 < x1 < 781:
                    drawColor = (231, 50, 176)
                elif 841 < x1 < 941:
                    drawColor = (255, 139, 232)
                elif 1001 < x1 < 1101:
                    drawColor = (104, 104, 104)
                elif 1161 < x1 < 1261:
                    drawColor = (0, 0, 0)
            ##Rectangulo que permite ver que color esta seleccionado
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Si solo dedo indice está levantado
        if fingers[1] and not fingers[2]:
            ##Circulo que permite ver la posicion del puntero
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Modo Dibujo")
            ##Condicional para que el dibujo inicie en la posicion del cursor y no en 0,0
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, grosordebrocha)
            ##Cuando esta seleccionado el borrador
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, grosordeborrador)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, grosordeborrador)
            ##Cuado se selecciona algun color
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, grosordebrocha)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, grosordebrocha)

            xp, yp = x1, y1
         #signo de ok
        if not fingers[0] and not fingers[1]:  
            current_time = time.time()
            #Se usa el tiempo para limitar la cantidad de capturas por segundo
            if current_time - tiempoultimacaptura >= tiempoentrecapturas:
                cv2.imwrite(f"captura_pantalla_{int(current_time)}.png", imgCanvas)
                print("Captura de pantalla tomada")
                tiempoultimacaptura = current_time

##Superponer el lienzo sobre la captura de video
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) ##Se convierte el lienzo del dibujo a una escala de grises
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) #
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header ##Posicionando el header en lienzo img
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)    ##LIenzo negro
    #cv2.imshow("Inv", imgInv)          ##Liengo
    cv2.waitKey(1)
