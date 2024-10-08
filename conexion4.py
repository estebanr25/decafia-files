from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import os
import subprocess
import time

app = Flask(__name__)

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'GET':
        # Código para enviar los tres mensajes de texto
        # Supongamos que los mensajes están almacenados en una lista llamada 'messages'
        with open("/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/afectaciones.txt", "r") as file:
            contenido = file.read()
        return contenido
    elif request.method == 'POST':
        # Obtener la imagen codificada en Base64 desde la solicitud
        base64_img = request.form['image']
        # Decodificar la imagen Base64
        img_data = base64.b64decode(base64_img)

        # Convertir la imagen decodificada a formato de imagen
        img = Image.open(io.BytesIO(img_data))

        # Guardar la imagen decodificada en una dirección específica
        img.save('/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/imagen_decodificada.jpg')
        time.sleep(0.5)
        # Ejecutar un archivo .py en otra dirección utilizando Python 3
        script_path = '/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/incefence_cpu4.py'
        process=subprocess.Popen(['python3', script_path])  # Ejecutar el archivo .py con Python 3
        # Espera a que el proceso termine y captura la salida
        stdout, stderr = process.communicate()

        # Imprime la salida del proceso (opcional)
        # Codificar la imagen decodificada en Base64
        with open('/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/envio_imagen.jpg', 'rb') as f:
            img_bw_binary = f.read()
        base64_img_bw = base64.b64encode(img_bw_binary).decode('utf-8')

        return base64_img_bw
        
@app.route('/upload_another_image', methods=['POST'])
def upload_another_image():
    if request.method == 'POST':
        with open('/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/overlay.jpg', 'rb') as f:
            img_bw_binary = f.read()
        base64_img_bw = base64.b64encode(img_bw_binary).decode('utf-8')

        return base64_img_bw
    return "OK"

@app.route('/mensaje', methods=['POST'])
def upload_another_image2():
    if request.method == 'POST':
        with open("/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/afectaciones.txt", "r") as file:
            contenido = file.read()
        return contenido

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
