import sys
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import mrcnn.visualize
import cv2
import os
import tensorflow as tf  # Importar TensorFlow
from PIL import Image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Configurar para usar solo CPU

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'HOJAS', 'ROYA', 'COCO', 'MINADOR']


def redimensionar_y_devolver(imagen_path, nuevo_ancho, nuevo_alto):
    # Abrir la imagen
    imagen = Image.open(imagen_path)
    # Redimensionar la imagen
    imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)  #Image.BILINEAR Image.LANCZOS
    return imagen_redimensionada

def calcular_nueva_dimension(imagen_path, objetivo_dim=960):
    # Abrir la imagen
    imagen = Image.open(imagen_path)
    # Obtener las dimensiones originales
    ancho_original, alto_original = imagen.size
    # Calcular nueva dimensión para que ya sea el ancho o el largo sea igual a objetivo_dim
    if ancho_original > alto_original:
        nuevo_ancho = objetivo_dim
        nuevo_alto = round((objetivo_dim / ancho_original) * alto_original)
    else:
        nuevo_alto = objetivo_dim
        nuevo_ancho = round((objetivo_dim / alto_original) * ancho_original)

    # Calcular el múltiplo de 64 más cercano
    nuevo_ancho = round(nuevo_ancho / 64) * 64
    nuevo_alto = round(nuevo_alto / 64) * 64
    return nuevo_ancho, nuevo_alto


def save_detection_plot(image, output_file, class_ids, scores, rois, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    # Diccionario para asignar un color único a cada clase
    color_dict = {
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'orange',  # Añade más colores según sea necesario
        # Agrega más clases y colores según sea necesario
    }

    # Mostrar cuadros delimitadores y etiquetas de todas las clases detectadas
    for class_id, score, bbox in zip(class_ids, scores, rois):
        if class_id in color_dict:  # Verifica si la clase tiene un color asignado
            y1, x1, y2, x2 = bbox
            label = class_names[class_id]
            color = color_dict[class_id]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1, f"{label} {score:.2f}", color='white', fontsize=6, bbox=dict(facecolor=color, alpha=0.5))

    plt.axis('off')
    plt.tight_layout()  # Ajustar el diseño de la figura para eliminar espacios en blanco no deseados
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)  # Guardar la figura en formato JPG
    plt.close()

def process_results(MASK, id_classs):
    largest_mask_index = None
    largest_mask_area = 0
    for i in range(MASK.shape[2]):
        if id_classs[i] == 1:
            mask_area = np.sum(MASK[:,:,i])
            if mask_area > largest_mask_area:
                largest_mask_area = mask_area
                largest_mask_index = i

    acumulate_2 = np.zeros_like(MASK[:,:,0], dtype=np.uint8)
    acumulate_3 = np.zeros_like(MASK[:,:,0], dtype=np.uint8)
    acumulate_4 = np.zeros_like(MASK[:,:,0], dtype=np.uint8)

    largest_mask = None
    if largest_mask_index is not None:
        largest_mask = MASK[:,:,largest_mask_index]

        for i in range(MASK.shape[2]):
            if id_classs[i] == 2:
                acumulate_2 = np.where(MASK[:,:,i], 1, acumulate_2)
            elif id_classs[i] == 3:
                acumulate_3 = np.where(MASK[:,:,i], 1, acumulate_3)
            elif id_classs[i] == 4:
                acumulate_4 = np.where(MASK[:,:,i], 1, acumulate_4)

    acumulate_2 = np.logical_and(largest_mask, acumulate_2)
    acumulate_3 = np.logical_and(largest_mask, acumulate_3)
    acumulate_4 = np.logical_and(largest_mask, acumulate_4)


    return largest_mask, acumulate_2, acumulate_3, acumulate_4

class CustomConfig(Config):
    # Give the configuration a recognizable name
    NAME="object"
    GPU_COUNT=1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # background + objetos
    IMAGE_MIN_DIM = 192
    IMAGE_MAX_DIM = 960
    STEPS_PER_EPOCH = 1934 #1934
    VALIDATION_STEPS = 20
    BACKBONE = 'resnet101' #'resnet50'
    MEAN_PIXEL = np.array([82.8, 124.3, 99.7]) # Media RGB: [ 82.7938383  124.32991352  99.68951586]
    RPN_ANCHOR_SCALES = (24,  38,  80, 480, 880)
    RPN_ANCHOR_RATIOS = [0.75, 1.0, 2.25]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]#
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005#0.0001
    RPN_NMS_THRESHOLD = 0.75
    DETECTION_MIN_CONFIDENCE = 0.79
    DETECTION_NMS_THRESHOLD = 0.3
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (60, 60)
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 78
    POST_NMS_ROIS_INFERENCE = 2000
    POST_NMS_ROIS_TRAINING = 1000
    DETECTION_MAX_INSTANCES = 78
    ROI_POSITIVE_RATIO = 0.3
    TRAIN_BN = True
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

config = CustomConfig()

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.89
inference_config = InferenceConfig()

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
with tf.device('/CPU:0'):  # Force Tensorflow to use CPU
    #model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(), model_dir=os.getcwd())
    ROOT_DIR = '/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT'
    assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

    sys.path.append(ROOT_DIR)
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    model = modellib.MaskRCNN(mode="inference", config=inference_config,  model_dir=MODEL_DIR)

    # Load the weights into the model.
    # Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    model.load_weights(filepath="mask_rcnn_960_epoc4.h5",
                       by_name=True)

    # load the input image, convert it from BGR to RGB channel
    #image = cv2.imread("/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/imagen_decodificada.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Nuevo ancho y alto calculado
    dir_image="/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/imagen_decodificada.jpg"
    nuevo_ancho, nuevo_alto = calcular_nueva_dimension(dir_image)
    # Redimensionar y devolver la imagen
    image = redimensionar_y_devolver(dir_image, nuevo_ancho, nuevo_alto)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Convertir de nuevo a formato PIL (RGB) para mostrar o guardar
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=0)

    # Get the results for the first image.
    r = r[0]
    largest_mask, acumulate_2, acumulate_3, acumulate_4 = process_results(r['masks'], r['class_ids'])
    # Superpone las máscaras acumuladas con colores diferentes
    overlay = np.zeros(largest_mask.shape + (3,), dtype=np.uint8)
    overlay[largest_mask == 1] = [255, 255, 0]  # Amarillo para clase 1
    overlay[acumulate_2 == 1] = [255, 0, 0]  # Rojo para clase 2
    overlay[acumulate_3 == 1] = [0, 255, 0]  # Verde para clase 3
    overlay[acumulate_4 == 1] = [0, 0, 255]  # Azul para clase 4
    cv2.imwrite("/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images/overlay.jpg", overlay)

    # Calcula el área de cada acumulación
    area_acumulate_2 = np.sum(acumulate_2)
    area_acumulate_3 = np.sum(acumulate_3)
    area_acumulate_4 = np.sum(acumulate_4)

    # Calcula la relación de área y expresa como porcentaje
    ratio_2 = np.round((area_acumulate_2 / np.sum(largest_mask)) * 100, 2)
    ratio_3 = np.round((area_acumulate_3 / np.sum(largest_mask)) * 100, 2)
    ratio_4 = np.round((area_acumulate_4 / np.sum(largest_mask)) * 100, 2)
    print(f' Roya: {ratio_2} Coco: {ratio_3}% Minador: {ratio_4}%')



    print(f"Relación de área para clase 2: {ratio_2:.2f}%")
    print(f"Relación de área para clase 3: {ratio_3:.2f}%")
    print(f"Relación de área para clase 4: {ratio_4:.2f}%")

   # result_image = mrcnn.visualize.display_instances2(image=image,
   #                               boxes=r['rois'],
   #                               masks=r['masks'],
    #                              class_ids=r['class_ids'],
     #                             class_names=CLASS_NAMES,
      #                            scores=r['scores'])

    # Directorio donde quieres guardar la imagen
    output_dir = "/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/images"
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio si no existe

    # Nombre del archivo de la imagen
    file_name = "envio_imagen.jpg"

    # Ruta completa del archivo
    file_path = os.path.join(output_dir, file_name)

    # Guardar la imagen en disco
    #with open(file_path, "wb") as f:
    #    f.write(result_image.getvalue())

    save_detection_plot(image, file_path, r['class_ids'], r['scores'], r['rois'], CLASS_NAMES)

    print("La imagen ha sido guardada en:", file_path)
    # Escribe las relaciones de área en el archivo "afectaciones.txt"
    with open("/home/estebanrosasusa/mask_rcnn_project/DECAFIA-MASK_RCNN_TF2_14_EDIT/afectaciones.txt", "w") as file:
        file.write(f' Roya: {ratio_2} Coco: {ratio_3}% Minador: {ratio_4}%')

