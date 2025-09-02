"""
Módulo de inicialización para el paquete de procesamiento de imágenes y análisis de círculos.

Este archivo permite importar los diferentes módulos desde el paquete principal.
Ejemplos de importación:

from modules import Filtros
from modules import OperacionesGeometricas
from modules import AnalizadorCirculos
"""

from .filtros import Filtros
from .operaciones_geometricas import OperacionesGeometricas
from .operaciones_morfologicas import OperacionesMorfologicas
from .analisis_circulos import AnalizadorCirculos
from .generador_reportes import GeneradorPDF
from .generador_pruebas import generar_imagenes_prueba, analizar_diferentes_formatos, generar_imagen_con_circulos

# Versión del paquete
__version__ = '1.0.0'
