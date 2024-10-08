# -*- coding: utf-8 -*-
"""Aumento de datos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18mL06G_vWp9jOoDjiJBn4GoxFs19pJfD
"""

import cv2
import numpy as np
import os
import random
from google.colab import drive
from google.colab.patches import cv2_imshow


# Montar Google Drive
drive.mount('/content/drive')

# Ruta de la carpeta con las imágenes originales en Google Drive
carpeta_originales = '/content/drive/MyDrive/NUEVO DATASET/SANAS'

# Ruta donde guardar las imágenes aumentadas en Google Drive
carpeta_aumentadas = '/content/drive/MyDrive/NUEVO DATASET/sanas aumentadas'

# Obtener la lista de archivos en la carpeta original
archivos = os.listdir(carpeta_originales)

# Elegir aleatoriamente la mitad de las imágenes
num_imagenes_aumentar = len(archivos) // 2
archivos_aumentar = random.sample(archivos, num_imagenes_aumentar)

# Iterar sobre cada imagen
for archivo in archivos:
    # Cargar la imagen original desde Google Drive
    ruta_original = os.path.join(carpeta_originales, archivo)
    imagen = cv2.imread(ruta_original)

    # Visualizar la imagen original
    #cv2_imshow(imagen)

    # Obtener dimensiones de la imagen original
    alto, ancho = imagen.shape[:2]

    # Calcular el factor de estiramiento para el lado más corto
    if alto < ancho:
        estiramiento_factor = random.uniform(1.4, 1.5)
    else:
        estiramiento_factor = random.uniform(1.4, 1.5)

    # Estirar la imagen en función del lado más corto
    if alto < ancho:
        nuevo_alto = int(alto * estiramiento_factor)
        nuevo_ancho = ancho
    else:
        nuevo_alto = alto
        nuevo_ancho = int(ancho * estiramiento_factor)

    imagen = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR) #interpolacion bilineal

    # Visualizar la imagen estirada
    #cv2_imshow(imagen)

    # Aplicar transformaciones aleatorias solo si la imagen está en la lista para aumentar
    if archivo in archivos_aumentar:
        # Reflejo horizontal
        if random.random() < 0.5:
            imagen = cv2.flip(imagen, 1)

        # Reflejo vertical
        if random.random() < 0.5:
            imagen = cv2.flip(imagen, 0)

        # Rotación aleatoria (izquierda o derecha)
        if random.random() < 0.5:
            imagen = cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)

        # Cambios aleatorios en la luminosidad, contraste y tonos de color
        if random.random() < 0.5:
            # Cambio aleatorio en la luminosidad
            alpha = 1.0 + random.uniform(-0.3, 0.1)
            # Cambio aleatorio en el contraste
            beta = 0.0 + random.uniform(-30, 30)
            # Aplicar cambios en la imagen
            imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

    # Visualizar la imagen aumentada
    # cv2_imshow(imagen)

    # Guardar la imagen aumentada en Google Drive
    nombre_archivo_aumentado = f"{os.path.splitext(archivo)[0]}_aumentado.jpg"
    ruta_aumentada = os.path.join(carpeta_aumentadas, nombre_archivo_aumentado)
    # Guardar la imagen aumentada (sin comentarios)
    cv2.imwrite(ruta_aumentada, imagen)