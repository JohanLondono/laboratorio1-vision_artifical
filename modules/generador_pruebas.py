import numpy as np
import cv2
import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image

def generar_imagen_con_circulos(ancho=640, alto=480, num_circulos=5, ruido=0.0, fondo=(255, 255, 255),
                               min_radio=20, max_radio=50):
    """
    Genera una imagen con círculos aleatorios.
    
    Args:
        ancho: Ancho de la imagen en píxeles
        alto: Alto de la imagen en píxeles
        num_circulos: Número de círculos a generar
        ruido: Nivel de ruido (0.0 a 1.0)
        fondo: Color de fondo (B, G, R)
        min_radio: Radio mínimo de los círculos
        max_radio: Radio máximo de los círculos
        
    Returns:
        Imagen generada con círculos
    """
    # Crear imagen en blanco
    imagen = np.ones((alto, ancho, 3), dtype=np.uint8)
    imagen[:] = fondo
    
    # Lista para guardar propiedades de los círculos
    circulos = []
    
    # Generar círculos aleatorios
    for _ in range(num_circulos):
        # Generar radio aleatorio
        radio = random.randint(min_radio, max_radio)
        
        # Generar centro aleatorio (evitando que el círculo se salga de la imagen)
        centro_x = random.randint(radio, ancho - radio)
        centro_y = random.randint(radio, alto - radio)
        
        # Verificar que no se solape demasiado con otros círculos
        solapamiento = False
        for c_x, c_y, r in circulos:
            # Calcular distancia entre centros
            dist = np.sqrt((centro_x - c_x)**2 + (centro_y - c_y)**2)
            # Si la distancia es menor que la suma de los radios, hay solapamiento
            if dist < (radio + r) * 0.8:
                solapamiento = True
                break
                
        if solapamiento:
            continue  # Intentar con otro círculo
            
        # Color aleatorio para el círculo
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Dibujar círculo
        cv2.circle(imagen, (centro_x, centro_y), radio, color, -1)
        
        # Guardar propiedades del círculo
        circulos.append((centro_x, centro_y, radio))
    
    # Añadir ruido si se especificó
    if ruido > 0:
        # Ruido gaussiano
        noise = np.random.normal(0, ruido * 255, imagen.shape).astype(np.int16)
        imagen = np.clip(imagen.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return imagen, circulos

def guardar_imagen_con_formato(imagen, ruta_base, formato='jpg', calidad=95):
    """
    Guarda una imagen en el formato especificado.
    
    Args:
        imagen: Imagen a guardar
        ruta_base: Ruta base para el archivo (sin extensión)
        formato: Formato de imagen (jpg, png, bmp)
        calidad: Calidad de compresión para JPG
        
    Returns:
        Ruta completa del archivo guardado
    """
    # Convertir imagen BGR a RGB para PIL
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    else:
        imagen_rgb = imagen
    
    # Crear objeto PIL
    pil_img = Image.fromarray(imagen_rgb)
    
    # Ruta completa con formato
    ruta_completa = f"{ruta_base}.{formato.lower()}"
    
    # Guardar según el formato
    if formato.lower() == 'jpg' or formato.lower() == 'jpeg':
        pil_img.save(ruta_completa, quality=calidad)
    else:
        pil_img.save(ruta_completa)
        
    return ruta_completa

def generar_imagenes_prueba(carpeta_destino='images', cantidad=3):
    """
    Genera un conjunto de imágenes de prueba con círculos.
    
    Args:
        carpeta_destino: Carpeta donde guardar las imágenes
        cantidad: Número de imágenes a generar
        
    Returns:
        Lista con las rutas de las imágenes generadas
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        
    rutas_imagenes = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(cantidad):
        # Parámetros aleatorios para cada imagen
        ancho = random.choice([640, 800, 1024])
        alto = int(ancho * 3/4)
        num_circulos = random.randint(3, 10)
        nivel_ruido = random.uniform(0, 0.1)
        fondo = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        
        # Generar imagen
        imagen, circulos = generar_imagen_con_circulos(
            ancho=ancho, 
            alto=alto,
            num_circulos=num_circulos,
            ruido=nivel_ruido,
            fondo=fondo
        )
        
        # Nombre base para la imagen
        nombre_base = os.path.join(carpeta_destino, f"circulo_{timestamp}_{i+1}")
        
        # Guardar en formato JPG
        ruta_jpg = guardar_imagen_con_formato(imagen, nombre_base, 'jpg')
        rutas_imagenes.append(ruta_jpg)
        
        # Para la primera imagen, guardar también en otros formatos
        if i == 0:
            guardar_imagen_con_formato(imagen, f"{nombre_base}_formato_png", 'png')
            guardar_imagen_con_formato(imagen, f"{nombre_base}_formato_bmp", 'bmp')
            
            # Guardar versiones con diferente calidad
            guardar_imagen_con_formato(imagen, f"{nombre_base}_calidad_baja", 'jpg', 25)
            guardar_imagen_con_formato(imagen, f"{nombre_base}_calidad_media", 'jpg', 50)
            
            # Guardar versiones con diferentes tamaños
            for escala, sufijo in [(0.5, '320x240'), (0.25, '160x120')]:
                img_redim = cv2.resize(
                    imagen, 
                    (int(ancho * escala), int(alto * escala)), 
                    interpolation=cv2.INTER_AREA
                )
                guardar_imagen_con_formato(img_redim, f"{nombre_base}_tamano_{sufijo}", 'jpg')
    
    return rutas_imagenes

def analizar_diferentes_formatos(analizador, ruta_imagen, carpeta_destino='images'):
    """
    Genera variaciones de una imagen en diferentes formatos y tamaños.
    
    Args:
        analizador: Instancia de AnalizadorCirculos o None
        ruta_imagen: Ruta a la imagen de referencia
        carpeta_destino: Carpeta donde guardar las imágenes generadas
        
    Returns:
        Lista con las rutas de las imágenes generadas
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        
    rutas_imagenes = []
    
    # Cargar la imagen original
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return rutas_imagenes
    
    # Obtener nombre base para la imagen
    nombre_original = os.path.basename(ruta_imagen)
    nombre_base, _ = os.path.splitext(nombre_original)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Generar versiones con diferentes formatos
    for formato, calidad in [('jpg', 95), ('jpg', 50), ('jpg', 25), ('png', 0), ('bmp', 0)]:
        nombre = f"{nombre_base}_formato_{formato}"
        if formato == 'jpg' and calidad != 95:
            nombre = f"{nombre_base}_formato_{formato}_calidad_{calidad}"
            
        ruta = os.path.join(carpeta_destino, f"{nombre}.{formato}")
        
        # Convertir y guardar
        if formato == 'jpg' or formato == 'jpeg':
            cv2.imwrite(ruta, imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        elif formato == 'png':
            cv2.imwrite(ruta, imagen, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(ruta, imagen)
            
        rutas_imagenes.append(ruta)
    
    # 2. Generar versiones con diferentes tamaños
    ancho_original, alto_original = imagen.shape[1], imagen.shape[0]
    for escala, sufijo in [(0.5, f"{int(ancho_original*0.5)}x{int(alto_original*0.5)}"), 
                          (0.25, f"{int(ancho_original*0.25)}x{int(alto_original*0.25)}"),
                          (2.0, f"{int(ancho_original*2.0)}x{int(alto_original*2.0)}")]:
        
        # Redimensionar imagen
        img_redim = cv2.resize(
            imagen, 
            (int(ancho_original * escala), int(alto_original * escala)), 
            interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
        )
        
        # Guardar imagen redimensionada
        nombre = f"{nombre_base}_tamano_{sufijo}"
        ruta = os.path.join(carpeta_destino, f"{nombre}.jpg")
        cv2.imwrite(ruta, img_redim, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rutas_imagenes.append(ruta)
    
    return rutas_imagenes
