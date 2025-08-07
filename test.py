import cv2
import numpy as np

# Cargar detector de caras
detector_cara = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Leer imagen
imagen = cv2.imread("merentiel-selfie.jpg")
if imagen is None:
    print("No se encontró la imagen.")
    exit()

# Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detección de cara
caras = detector_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

# Desenfocar toda la imagen
desenfocada = cv2.GaussianBlur(imagen, (55, 55), 0)


# Crear máscara para suavizar el borde del rostro
mascara = np.zeros(imagen.shape[:2], dtype="uint8")

for (x, y, w, h) in caras:
    # Coordenadas del centro del rostro
    centro_x = x + w // 2
    centro_y = y + h // 2
    radio = max(w, h) // 2

    # Dibujar círculo blanco en la zona del rostro
    cv2.circle(mascara, (centro_x, centro_y), radio, 255, -1)

# Suavizar los bordes de la máscara
mascara_suave = cv2.GaussianBlur(mascara, (35, 35), 0)

# Normalizar máscara a [0,1]
mascara_suave = mascara_suave.astype(float) / 255

# Expandir canales
mascara_suave = cv2.merge([mascara_suave]*3)

# Aplicar máscara a mezcla de imagen original y desenfocada
resultado = (imagen * mascara_suave + desenfocada * (1 - mascara_suave)).astype(np.uint8)

# Mostrar resultado
cv2.imshow("Modo retrato", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
