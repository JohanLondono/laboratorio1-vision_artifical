# Análisis y Detección de Círculos en Imágenes

Este proyecto implementa un sistema modular para la detección y análisis de círculos en imágenes utilizando técnicas avanzadas de procesamiento digital. La arquitectura modular permite un mantenimiento más sencillo y una mejor organización del código.

## Requisitos

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Seaborn
- ReportLab (para generación de PDF)
- Pillow

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
python/
├── images/              # Directorio para las imágenes de entrada
├── resultados/          # Directorio para los resultados y análisis
├── modules/             # Módulos específicos para cada funcionalidad
│   ├── __init__.py      # Inicialización de paquete
│   ├── analisis_circulos.py    # Análisis y detección de círculos
│   ├── filtros.py              # Operaciones de filtrado de imágenes
│   ├── operaciones_geometricas.py  # Transformaciones geométricas
│   ├── operaciones_morfologicas.py # Operaciones morfológicas
│   ├── generador_reportes.py   # Generación de reportes PDF
│   ├── generador_pruebas.py    # Generación de imágenes de prueba
├── main.py              # Script principal con menú interactivo
```

## Ejecución

Para ejecutar la aplicación principal con menú interactivo:

```bash
python main.py
```

Opciones disponibles:

- `--dir-imagenes`: Directorio donde se encuentran o se generarán las imágenes (por defecto: 'images')
- `--dir-resultados`: Directorio donde se guardarán los resultados (por defecto: 'resultados')

Ejemplo:

```bash
python main.py --dir-imagenes "mis_imagenes" --dir-resultados "mis_resultados"
```

Para generar documentación a partir de resultados existentes:

```bash
python generar_documentacion.py
```

## Módulos del Sistema

### 1. Módulo de Filtros (`filtros.py`)
Implementa diferentes filtros para el procesamiento de imágenes:
- Desenfoque (Blur)
- Filtro gaussiano
- Filtro de mediana
- Filtro de nitidez
- Filtro bilateral
- Detección de bordes (Canny)
- Ecualización de histograma

### 2. Módulo de Operaciones Geométricas (`operaciones_geometricas.py`)
Proporciona operaciones de transformación geométrica:
- Redimensionamiento de imágenes
- Rotación
- Recorte
- Volteo horizontal/vertical
- Traslación
- Transformación de perspectiva

### 3. Módulo de Operaciones Morfológicas (`operaciones_morfologicas.py`)
Implementa operaciones morfológicas para imágenes binarias:
- Erosión
- Dilatación
- Apertura
- Cierre
- Gradiente morfológico
- Top Hat
- Black Hat

### 4. Módulo de Análisis de Círculos (`analisis_circulos.py`)
Contiene algoritmos para la detección y análisis de círculos:

#### 4.1 Transformada de Hough para Círculos
Implementa la función `HoughCircles` de OpenCV para detectar círculos en imágenes:
- **Fundamento Matemático**: La transformada de Hough mapea puntos de la imagen a un espacio de parámetros tridimensional (centro_x, centro_y, radio).
- **Proceso**:
  1. Detección de bordes (típicamente con Canny)
  2. Acumulación de votos en un espacio de parámetros 3D
  3. Identificación de máximos locales como círculos
- **Parámetros Clave**:
  - `dp`: Ratio de resolución del acumulador respecto a la imagen
  - `minDist`: Distancia mínima entre centros de círculos detectados
  - `param1`: Umbral superior para el detector de bordes Canny
  - `param2`: Umbral para detección de centros (menor valor = más círculos falsos)
  - `minRadius/maxRadius`: Rango de radios a detectar

#### 4.2 Detección por Contornos y Análisis de Forma
Implementa un enfoque basado en contornos con análisis de circularidad:
- **Fundamento Matemático**: Un círculo perfecto tiene una circularidad de 1.0 (4π × área / perímetro²)
- **Proceso Mejorado**:
  1. Preprocesamiento avanzado (ecualización, umbralización adaptativa)
  2. Enfoque multi-umbral para combinar resultados de diferentes binarizaciones
  3. Operaciones morfológicas para mejorar contornos (cierre para unir bordes)
  4. Detección de contornos usando OpenCV
  5. Filtrado por área mínima y circularidad
  6. Recuperación de círculos parciales mediante umbrales adaptados
  7. Eliminación de detecciones duplicadas
- **Ventajas de la versión mejorada**:
  - Mayor robustez ante variaciones de iluminación
  - Mejor detección en imágenes con círculos parciales o superpuestos
  - Capacidad para detectar círculos en diferentes condiciones de imagen

### 5. Módulo de Generación de Reportes (`generador_reportes.py`)
Permite crear informes detallados en formato PDF:
- Generación de gráficas y visualizaciones
- Inclusión de estadísticas y tablas
- Exportación a PDF con formato profesional

### 6. Módulo de Generación de Imágenes de Prueba (`generador_pruebas.py`)
Crea imágenes sintéticas para pruebas:
- Círculos de diferentes tamaños y posiciones
- Diferentes formatos (JPG, PNG, BMP)
- Distintos niveles de compresión y tamaños

## Características Principales

1. **Interfaz por Menú**: Interfaz de línea de comandos con menús interactivos
2. **Procesamiento Completo**: Aplicación de múltiples técnicas de procesamiento de imágenes
3. **Análisis Detallado**: Métricas y estadísticas de los círculos detectados
4. **Generación de Reportes**: Creación de informes profesionales en PDF
5. **Documentación Automática**: Análisis comparativo de métodos y resultados
6. **Estructura Modular**: Organización que facilita la extensibilidad y mantenimiento

## Pipeline de Procesamiento de Imágenes

El sistema implementa la siguiente secuencia de procesamiento para detectar círculos:

1. **Carga de imagen**: Lectura del archivo y conversión a formato adecuado
2. **Conversión a escala de grises**: Eliminación del canal de color para simplificar el análisis
3. **Filtrado**: Aplicación de filtro Gaussiano para reducir ruido manteniendo bordes
4. **Mejora de contraste**: Ecualización de histograma para normalizar la distribución de intensidades
5. **Binarización**: 
   - Para Hough: Umbralización simple
   - Para Contornos: Umbralización adaptativa con múltiples valores (multi-umbral)
6. **Operaciones morfológicas**: Aplicación de cierre seguido de apertura para:
   - Conectar bordes parciales de círculos
   - Eliminar pequeños ruidos y artefactos
7. **Detección**: Aplicación del método seleccionado (Hough o Contornos)
8. **Post-procesamiento**: Filtrado y validación de resultados
9. **Análisis y cálculo de métricas**: Obtención de estadísticas sobre los círculos detectados

## Resultados

El programa genera varios tipos de resultados:

1. **Imágenes Procesadas**: Imágenes en distintas etapas del procesamiento
2. **Resultados de Detección**: Imágenes con círculos detectados y marcados
3. **Datos Estadísticos**: Archivos Excel con métricas detalladas
4. **Informes PDF**: Reportes profesionales con los hallazgos del análisis
5. **Documentación**: Análisis comparativos y conclusiones

## Análisis y Conclusiones

El proyecto permite evaluar y comparar:

1. **Eficacia de los Métodos de Detección**:
   - La transformada de Hough es más precisa pero más lenta
   - El método de contornos mejorado proporciona:
     - Mayor robustez ante diferentes condiciones de iluminación
     - Mejor detección en imágenes con círculos parciales
     - Mayor eficiencia computacional para imágenes grandes

2. **Efecto del Formato de Imagen**:
   - Los formatos sin pérdida preservan mejor los detalles
   - La compresión afecta significativamente la detección

3. **Efecto del Tamaño de la Imagen**:
   - Mayor resolución permite detectar círculos más pequeños
   - Existe un balance óptimo entre precisión y tiempo de procesamiento

## Autor

Desarrollado para el curso de Visión Artificial
