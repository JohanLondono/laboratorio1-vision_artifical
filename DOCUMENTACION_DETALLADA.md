# Documentación Detallada: Sistema Modular de Análisis y Detección de Círculos

## 1. Introducción

Este documento describe en detalle el sistema modular implementado para la detección y análisis de círculos en imágenes utilizando técnicas avanzadas de procesamiento digital. El sistema ha sido diseñado con una arquitectura modular que separa las distintas funcionalidades en módulos especializados, lo que facilita su mantenimiento, extensión y comprensión.

El objetivo principal del sistema es aplicar y evaluar diferentes técnicas de procesamiento de imágenes para detectar formas circulares, medir sus características geométricas (área, perímetro, radio) y analizar cómo afectan diferentes formatos y tamaños de imagen a la precisión de la detección.

## 2. Arquitectura del Sistema

El sistema está organizado en un conjunto de módulos especializados que interactúan entre sí para proporcionar la funcionalidad completa. Esta arquitectura modular permite la separación de responsabilidades y facilita la extensión del sistema.

### 2.1 Estructura de Directorios

```
python/
├── images/                 # Imágenes de entrada
├── resultados/             # Resultados generados
│   └── documentacion/      # Documentación autogenerada
├── modules/                # Módulos específicos
│   ├── __init__.py         # Inicialización del paquete
│   ├── analisis_circulos.py       # Análisis de círculos
│   ├── filtros.py                 # Filtros de imagen
│   ├── operaciones_geometricas.py # Operaciones geométricas
│   ├── operaciones_morfologicas.py # Operaciones morfológicas
│   ├── generador_reportes.py      # Generación de reportes PDF
│   └── generador_pruebas.py       # Imágenes de prueba
├── main.py                 # Script principal
```

### 2.2 Diagrama de Componentes

```
┌─────────────────┐      ┌────────────────────┐
│     Main.py     │◄────►│  AnalizadorCirculos│
└────────┬────────┘      └─────────┬──────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐      ┌────────────────────┐
│    Filtros      │◄────►│OperacionesGeometricas
└─────────────────┘      └────────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐      ┌────────────────────┐
│OperacionesMorfologicas│ GeneradorReportes   │
└─────────────────┘      └────────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐      ┌────────────────────┐
│GeneradorPruebas │      │GenerarDocumentacion│
└─────────────────┘      └────────────────────┘
```

## 3. Descripción de los Módulos

### 3.1 Módulo de Filtros (`filtros.py`)

Este módulo implementa diversas operaciones de filtrado para el procesamiento de imágenes:

#### 3.1.1 Funcionalidades Principales

- **Filtro de Desenfoque**: Aplica un desenfoque promedio a la imagen.
  ```python
  def aplicar_filtro_desenfoque(imagen, kernel_size=(5, 5)):
      return cv2.blur(imagen, kernel_size)
  ```

- **Filtro Gaussiano**: Aplica un desenfoque gaussiano que preserva mejor los bordes.
  ```python
  def aplicar_filtro_gaussiano(imagen, kernel_size=(5, 5), sigma=0):
      return cv2.GaussianBlur(imagen, kernel_size, sigma)
  ```

- **Filtro de Mediana**: Ideal para eliminar ruido "sal y pimienta" preservando bordes.
  ```python
  def aplicar_filtro_mediana(imagen, kernel_size=5):
      return cv2.medianBlur(imagen, kernel_size)
  ```

- **Filtro de Nitidez**: Aumenta el contraste en los bordes para resaltar detalles.
  ```python
  def aplicar_filtro_nitidez(imagen):
      kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      return cv2.filter2D(imagen, -1, kernel)
  ```

- **Filtro Bilateral**: Reduce ruido preservando bordes definidos.
  ```python
  def aplicar_filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75):
      return cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
  ```

- **Detector de Bordes Canny**: Identifica bordes en la imagen.
  ```python
  def detectar_bordes_canny(imagen, umbral1=100, umbral2=200):
      return cv2.Canny(imagen, umbral1, umbral2)
  ```

- **Ecualización de Histograma**: Mejora el contraste global de la imagen.
  ```python
  def ecualizar_histograma(imagen):
      return cv2.equalizeHist(imagen)
  ```

### 3.2 Módulo de Operaciones Geométricas (`operaciones_geometricas.py`)

Proporciona transformaciones geométricas para modificar la forma o posición de las imágenes:

#### 3.2.1 Funcionalidades Principales

- **Redimensionamiento**: Cambia el tamaño de la imagen, manteniendo o no la proporción.
  ```python
  def redimensionar_imagen(imagen, ancho=None, alto=None):
      # Cálculos para mantener proporción si se proporciona una sola dimensión
      return cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)
  ```

- **Rotación**: Rota la imagen un ángulo especificado alrededor de su centro.
  ```python
  def rotar_imagen(imagen, angulo):
      (h, w) = imagen.shape[:2]
      centro = (w // 2, h // 2)
      M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
      return cv2.warpAffine(imagen, M, (w, h))
  ```

- **Volteo**: Invierte la imagen horizontal, verticalmente o ambos.
  ```python
  def voltear_imagen(imagen, modo):
      return cv2.flip(imagen, modo)  # 0=vertical, 1=horizontal, -1=ambos
  ```

- **Traslación**: Mueve la imagen en el plano x,y.
  ```python
  def trasladar_imagen(imagen, dx, dy):
      M = np.float32([[1, 0, dx], [0, 1, dy]])
      return cv2.warpAffine(imagen, M, (w, h))
  ```

- **Recorte**: Extrae una región específica de la imagen.
  ```python
  def recortar_imagen(imagen, x1, y1, x2, y2):
      return imagen[y1:y2, x1:x2]
  ```

### 3.3 Módulo de Operaciones Morfológicas (`operaciones_morfologicas.py`)

Implementa operaciones morfológicas para imágenes binarias que modifican la estructura de los objetos:

#### 3.3.1 Funcionalidades Principales

- **Creación de Kernels**: Genera kernels con diferentes formas para operaciones morfológicas.
  ```python
  def crear_kernel(forma='rectangulo', tamano=5):
      if forma == 'rectangulo':
          return np.ones((tamano, tamano), np.uint8)
      elif forma == 'elipse':
          return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
      elif forma == 'cruz':
          return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamano, tamano))
  ```

- **Erosión**: Reduce regiones blancas, elimina objetos pequeños y detalles.
  ```python
  def erosion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
      kernel = crear_kernel(kernel_forma, kernel_size)
      return cv2.erode(imagen, kernel, iterations=iteraciones)
  ```

- **Dilatación**: Expande regiones blancas, une objetos cercanos y rellena huecos pequeños.
  ```python
  def dilatacion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
      kernel = crear_kernel(kernel_forma, kernel_size)
      return cv2.dilate(imagen, kernel, iterations=iteraciones)
  ```

- **Apertura**: Erosión seguida de dilatación, elimina ruido y detalles pequeños.
  ```python
  def apertura(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
      kernel = crear_kernel(kernel_forma, kernel_size)
      return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
  ```

- **Cierre**: Dilatación seguida de erosión, rellena huecos pequeños y une componentes cercanos.
  ```python
  def cierre(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
      kernel = crear_kernel(kernel_forma, kernel_size)
      return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
  ```

- **Operaciones Avanzadas**:
  - Gradiente morfológico: Diferencia entre dilatación y erosión (resalta bordes).
  - Top Hat: Diferencia entre imagen original y apertura (resalta detalles pequeños).
  - Black Hat: Diferencia entre cierre e imagen original (resalta huecos oscuros).

### 3.4 Módulo de Análisis de Círculos (`analisis_circulos.py`)

Este es el módulo central que coordina el proceso de detección y análisis de círculos en imágenes:

#### 3.4.1 Funcionalidades Principales

- **Carga y Preprocesamiento**: Carga imágenes y las prepara para el análisis.
  ```python
  def cargar_imagen(self, ruta_imagen):
      self.imagen_original = cv2.imread(ruta_imagen)
      self.imagen_rgb = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
      return self.imagen_rgb
      
  def convertir_escala_grises(self, imagen=None):
      if imagen is None:
          imagen = self.imagen_original
      self.imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
      return self.imagen_gris
  ```

- **Detección de Círculos por Transformada de Hough**: Método especializado para detección de círculos.
  ```python
  def detectar_circulos_hough(self, imagen=None, metodo=cv2.HOUGH_GRADIENT, 
                            dp=1, minDist=50, param1=50, param2=30, 
                            minRadius=0, maxRadius=0):
      # Implementación de la detección de círculos mediante HoughCircles
  ```

- **Detección de Círculos por Contornos**: Alternativa que analiza contornos y filtra por circularidad.
  ```python
  def detectar_circulos_contornos(self, imagen=None, metodo=cv2.RETR_EXTERNAL, 
                                aprox=cv2.CHAIN_APPROX_SIMPLE, min_area=100):
      # Implementación de detección usando contornos y análisis de circularidad
  ```

- **Análisis de Características**: Cálculo de propiedades geométricas de los círculos.
  ```python
  # Cálculos dentro de los métodos de detección:
  area = np.pi * radio ** 2
  perimetro = 2 * np.pi * radio
  self.areas.append(area)
  self.perimetros.append(perimetro)
  self.radios.append(radio)
  ```

- **Procesamiento de Múltiples Imágenes**: Automatización del procesamiento por lotes.
  ```python
  def procesar_multiples_imagenes(self, lista_rutas, metodo_deteccion='hough'):
      for ruta in lista_rutas:
          self.procesar_imagen(ruta, metodo_deteccion)
      return self.resultados_df
  ```

- **Visualización de Resultados**: Creación de visualizaciones para análisis cualitativo.
  ```python
  def visualizar_resultados(self, ruta_imagen=None, metodo_deteccion='hough'):
      # Genera una figura con múltiples subplots mostrando las diferentes
      # etapas del procesamiento y los círculos detectados
  ```

### 3.5 Módulo de Generación de Reportes (`generador_reportes.py`)

Permite crear informes detallados en formato PDF con los resultados del análisis:

#### 3.5.1 Funcionalidades Principales

- **Creación de Documentos PDF**: Genera documentos PDF estructurados.
  ```python
  def generar_informe(self, excel_path, titulo=None, autor=None, conclusiones=None, 
                     ruta_imagen=None, imagen_procesada=None):
      # Creación de un documento PDF con ReportLab
  ```

- **Inclusión de Imágenes**: Añade las imágenes originales y procesadas al informe.
  ```python
  # Guardar imágenes temporales para incluir en el PDF
  img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
  cv2.imwrite(img_temp.name, img)
  img_reportlab = Image(img_temp.name, width=4*inch, height=3*inch)
  story.append(img_reportlab)
  ```

- **Generación de Tablas y Gráficos**: Crea representaciones visuales de los datos.
  ```python
  # Crear una tabla con los resultados
  tabla_resumen = Table(data_resumen, colWidths=[3*inch, 2*inch])
  tabla_resumen.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
      ('GRID', (0, 0), (-1, -1), 1, colors.black),
      ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
  ]))
  ```

### 3.6 Módulo de Generación de Imágenes de Prueba (`generador_pruebas.py`)

Crea imágenes sintéticas para probar el sistema:

#### 3.6.1 Funcionalidades Principales

- **Generación de Imágenes con Círculos**: Crea imágenes con círculos aleatorios.
  ```python
  def generar_imagen_con_circulos(ancho=640, alto=480, num_circulos=5, ruido=0.0, 
                                fondo=(255, 255, 255), min_radio=20, max_radio=50):
      # Genera una imagen con círculos aleatorios
  ```

- **Variación de Formatos**: Guarda imágenes en diferentes formatos y calidades.
  ```python
  def guardar_imagen_con_formato(imagen, ruta_base, formato='jpg', calidad=95):
      # Guarda la imagen en el formato especificado con la calidad indicada
  ```

- **Creación de Conjuntos de Prueba**: Genera conjuntos de imágenes para análisis comparativo.
  ```python
  def generar_imagenes_prueba(carpeta_destino='images', cantidad=3):
      # Genera varias imágenes con diferentes parámetros
  ```

- **Análisis de Efecto de Formatos**: Genera variantes de una misma imagen para comparar.
  ```python
  def analizar_diferentes_formatos(analizador, ruta_imagen, carpeta_destino='images'):
      # Genera diferentes variantes de formato y tamaño para una imagen
  ```

### 3.7 Script Principal (`main.py`)

El punto de entrada principal del sistema que coordina todos los módulos:

#### 3.7.1 Funcionalidades Principales

- **Interfaz de Menú**: Proporciona un menú interactivo para utilizar todas las funcionalidades.
  ```python
  class MenuAplicacion:
      def __init__(self):
          # Inicializa los componentes
          self.filtros = Filtros()
          self.op_geometricas = OperacionesGeometricas()
          self.op_morfologicas = OperacionesMorfologicas()
          self.analizador = AnalizadorCirculos(self.dir_imagenes, self.dir_resultados)
      
      def mostrar_menu_principal(self):
          # Muestra el menú principal con opciones
  ```

- **Gestión de Imágenes**: Permite cargar, procesar y guardar imágenes.
  ```python
  def cargar_imagen(self, ruta_imagen):
      # Carga una imagen desde un archivo
      
  def guardar_imagen_procesada(self):
      # Guarda la imagen procesada actualmente
  ```

- **Coordinación de Procesamiento**: Coordina las diferentes etapas del procesamiento.
  ```python
  def detectar_circulos_hough(self):
      # Preprocesa la imagen y aplica la detección de círculos
      img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
      img_filtrada = self.filtros.aplicar_filtro_gaussiano(img_gris)
      imagen_resultado, circulos = self.analizador.detectar_circulos_hough(img_filtrada, ...)
  ```

- **Generación de Reportes y Documentación**: Automatiza la creación de informes.
  ```python
  def generar_informe_pdf(self):
      # Genera un informe PDF con los resultados del análisis
      
  def visualizar_resultados_circulos(self):
      # Crea visualizaciones gráficas de los resultados
  ```

## 4. Flujo de Trabajo Completo

El sistema implementa el siguiente flujo de trabajo para el análisis de imágenes:

### 4.1 Preprocesamiento de Imágenes

1. **Carga de Imagen**: La imagen se carga desde un archivo o se captura desde una cámara.
2. **Conversión a Escala de Grises**: Se elimina la información de color para simplificar el procesamiento.
3. **Aplicación de Filtros**: Se aplican filtros (generalmente gaussiano) para reducir el ruido.
4. **Binarización**: Se convierte la imagen a blanco y negro mediante umbralización.
5. **Operaciones Morfológicas**: Se aplican operaciones como cierre o apertura para mejorar la segmentación.

### 4.2 Detección de Círculos

Se implementan dos métodos principales de detección:

#### 4.2.1 Método de Transformada de Hough

1. La imagen preprocesada se pasa a la función `HoughCircles` de OpenCV.
2. Se configuran parámetros como resolución del acumulador, distancia entre círculos, umbrales y rangos de radio.
3. El algoritmo detecta círculos basándose en la transformación del espacio de la imagen al espacio de Hough.

#### 4.2.2 Método de Contornos

1. Se detectan contornos en la imagen binaria.
2. Para cada contorno, se calculan propiedades como área y perímetro.
3. Se calcula la circularidad (4π × área / perímetro²) para determinar si el contorno es un círculo.
4. Se filtran los contornos por circularidad y tamaño mínimo.

### 4.3 Análisis y Medición

Una vez detectados los círculos, se realizan las siguientes mediciones:

1. **Radio**: Se extrae directamente de la detección o se calcula a partir del contorno.
2. **Área**: Se calcula como π × r².
3. **Perímetro**: Se calcula como 2π × r o directamente del contorno.
4. **Distribución Estadística**: Se calculan estadísticas como media, mínimo, máximo y desviación estándar.

### 4.4 Generación de Resultados

Los resultados se presentan en múltiples formatos:

1. **Visualización en Pantalla**: Imágenes con círculos marcados y estadísticas básicas.
2. **Archivos Excel**: Tablas detalladas con todas las métricas calculadas.
3. **Informes PDF**: Documentos estructurados con análisis, gráficos y conclusiones.
4. **Documentación HTML**: Análisis comparativo con visualizaciones interactivas.

## 5. Evaluación y Comparación de Métodos

### 5.1 Comparativa de Métodos de Detección

| Aspecto | Transformada de Hough | Método de Contornos |
|---------|----------------------|---------------------|
| Precisión | Alta para círculos perfectos | Media, depende de la binarización |
| Velocidad | Más lenta | Más rápida |
| Robustez ante ruido | Media | Baja, requiere buen preprocesamiento |
| Detección de círculos parciales | Buena | Limitada |
| Personalización | Muchos parámetros | Menos parámetros |
| Complejidad computacional | Alta | Media |

### 5.2 Efecto del Formato de Imagen

| Formato | Ventajas | Desventajas | Impacto en la Detección |
|---------|---------|------------|------------------------|
| PNG | Sin pérdida, preserva detalles | Archivos más grandes | Excelente detección de bordes |
| BMP | Sin compresión, máxima calidad | Archivos muy grandes | Óptima detección |
| JPEG (alta calidad) | Buen balance tamaño/calidad | Ligera pérdida de información | Buena detección general |
| JPEG (baja calidad) | Archivos pequeños | Artefactos de compresión | Problemas con círculos pequeños |

### 5.3 Efecto del Tamaño de Imagen

| Tamaño | Ventajas | Desventajas | Impacto en la Detección |
|--------|---------|------------|------------------------|
| Grande (800x800+) | Alta precisión, detecta círculos pequeños | Mayor tiempo de procesamiento | Excelente pero lento |
| Medio (400x400) | Buen balance precisión/velocidad | Puede perder círculos muy pequeños | Adecuado para la mayoría de casos |
| Pequeño (200x200) | Procesamiento rápido | Pérdida de detalles finos | Problemas con círculos pequeños |

## 6. Conclusiones y Recomendaciones

### 6.1 Conclusiones Generales

1. **Efectividad de los Métodos**:
   - La transformada de Hough es más precisa para círculos bien definidos y puede detectar círculos parcialmente visibles.
   - El método de contornos es computacionalmente más eficiente pero sensible a la calidad de la binarización.

2. **Importancia del Preprocesamiento**:
   - La aplicación de filtros (especialmente gaussiano) mejora significativamente la detección.
   - Las operaciones morfológicas son cruciales para obtener buenos contornos, especialmente el cierre para rellenar discontinuidades.

3. **Efecto de los Parámetros**:
   - La selección adecuada de parámetros (umbral, tamaño de kernel, etc.) es crítica y depende de las características específicas de la imagen.
   - Se recomienda un enfoque adaptativo que ajuste los parámetros según las propiedades de cada imagen.

### 6.2 Recomendaciones Prácticas

1. **Para Imágenes Ruidosas**:
   - Aplicar filtrado gaussiano con kernel 5x5 o mayor
   - Usar operaciones morfológicas de cierre
   - Preferir transformada de Hough con parámetros ajustados

2. **Para Procesamiento en Tiempo Real**:
   - Redimensionar imágenes a tamaño medio (400x400)
   - Utilizar método de contornos
   - Optimizar el preprocesamiento (solo pasos esenciales)

3. **Para Máxima Precisión**:
   - Usar formatos sin pérdida (PNG, BMP)
   - Aplicar filtros de nitidez después del suavizado
   - Combinar ambos métodos de detección y comparar resultados

4. **Para Detección Robusta**:
   - Implementar un enfoque en dos fases: detección aproximada seguida de refinamiento
   - Utilizar información contextual (tamaño esperado, distribución)
   - Validar resultados mediante métricas de circularidad y consistencia

## 7. Referencias

1. OpenCV Documentation: https://docs.opencv.org/
2. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
3. Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
4. Bradski, G., & Kaehler, A. (2008). Learning OpenCV. O'Reilly Media.
5. Davies, E. R. (2017). Computer Vision: Principles, Algorithms, Applications, Learning. Academic Press.
