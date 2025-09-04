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

#### 3.4.2 Detección de Círculos por Transformada de Hough

La transformada de Hough para círculos es un algoritmo específico para detectar formas circulares en imágenes:

**Fundamento Matemático**: Un círculo puede representarse por la ecuación:
```
(x - a)² + (y - b)² = r²
```
Donde (a,b) es el centro y r es el radio.

**Implementación**:
```python
def detectar_circulos_hough(self, imagen=None, metodo=cv2.HOUGH_GRADIENT, 
                          dp=1, minDist=50, param1=50, param2=30, 
                          minRadius=0, maxRadius=0):
    # Implementación de la detección de círculos mediante HoughCircles
    # ...
    self.circulos = cv2.HoughCircles(
        imagen, 
        metodo, 
        dp, 
        minDist, 
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    # Procesamiento de los resultados...
```

**Proceso Interno de la Transformada de Hough para Círculos**:
1. **Preprocesamiento**: La imagen se convierte a escala de grises y se aplica un filtro para reducir ruido.
2. **Detección de Bordes**: Se aplica un detector de bordes (típicamente Canny) para identificar los píxeles de borde.
3. **Votación**: Para cada píxel de borde (x,y), se consideran todos los posibles círculos que podrían pasar por ese punto:
   - Para cada radio posible dentro del rango [minRadius, maxRadius]
   - Se incrementa un contador en el espacio de Hough (a,b,r)
4. **Acumulación**: Se construye un acumulador 3D donde cada celda (a,b,r) contiene el número de "votos" para un círculo con centro (a,b) y radio r.
5. **Identificación de Máximos**: Se buscan los máximos locales en el acumulador, que corresponden a los círculos más probables en la imagen.
6. **Filtrado**: Se aplican umbrales (param1, param2) y distancia mínima (minDist) para eliminar detecciones duplicadas o falsas.

**Significado de los Parámetros**:
- `dp`: Resolución inversa del acumulador. Si dp=1, el acumulador tiene la misma resolución que la imagen. Si dp=2, el acumulador tiene la mitad de resolución.
- `minDist`: Distancia mínima entre centros de círculos detectados. Si es demasiado pequeña, se detectarán círculos múltiples (falsos) donde solo hay uno.
- `param1`: Umbral superior para el detector de bordes Canny.
- `param2`: Umbral para la detección de centros. Un valor más bajo detectará más círculos (incluyendo falsos).
- `minRadius`, `maxRadius`: Límites del radio de los círculos a detectar.

#### 3.4.3 Detección de Círculos por Contornos (Método Mejorado)

La detección de círculos por contornos es un enfoque más flexible que se basa en el análisis de forma:

**Fundamento Matemático**: Un círculo perfecto tiene una circularidad (compacidad) igual a 1, calculada como:
```
Circularidad = 4π × Área / Perímetro²
```

**Implementación Mejorada**:
```python
def detectar_circulos_contornos(self, imagen=None, metodo=cv2.RETR_LIST, 
                              aprox=cv2.CHAIN_APPROX_SIMPLE, min_area=50, 
                              circularidad_min=0.6, invertir=False, debug=False,
                              multi_umbral=True, completar_circulos=True):
    # Preprocesamiento avanzado de la imagen
    # Implementación del enfoque multi-umbral
    # Detección y filtrado de contornos
    # Análisis de circularidad y recuperación de círculos parciales
    # ...
```

**Proceso Detallado del Método Mejorado**:

1. **Preprocesamiento Avanzado**:
   - **Ecualización de Histograma**: Normaliza la distribución de intensidades para mejorar el contraste.
   ```python
   imagen_ecualizada = cv2.equalizeHist(imagen)
   ```
   
   - **Detección de Bordes**: Aplica el algoritmo Canny para identificar bordes.
   ```python
   bordes = cv2.Canny(imagen_ecualizada, 50, 150)
   ```
   
   - **Dilatación de Bordes**: Expande los bordes para conectar discontinuidades.
   ```python
   kernel_dilate = np.ones((3, 3), np.uint8)
   bordes_dilatados = cv2.dilate(bordes, kernel_dilate, iterations=1)
   ```

2. **Enfoque Multi-umbral**:
   - Aplica múltiples niveles de umbralización para capturar círculos con diferentes intensidades:
   ```python
   imagenes_binarias = []
   for umbral in [50, 100, 150, 200]:
       _, img_bin = cv2.threshold(imagen_ecualizada, umbral, 255, cv2.THRESH_BINARY)
       imagenes_binarias.append(img_bin)
   ```
   
   - También aplica umbrales adaptativos que se ajustan a las condiciones locales:
   ```python
   img_adapt1 = cv2.adaptiveThreshold(imagen_ecualizada, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
   img_adapt2 = cv2.adaptiveThreshold(imagen_ecualizada, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
   ```
   
   - Umbral de Otsu para determinación automática del nivel óptimo:
   ```python
   _, img_otsu = cv2.threshold(imagen_ecualizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   ```
   
   - Combina todos los resultados mediante operación OR lógico:
   ```python
   imagen_combinada = np.zeros_like(imagen)
   for img_bin in imagenes_binarias:
       imagen_combinada = cv2.bitwise_or(imagen_combinada, img_bin)
   ```

3. **Operaciones Morfológicas**:
   - Aplica cierre para conectar segmentos próximos de contornos:
   ```python
   kernel = np.ones((5, 5), np.uint8)
   imagen_combinada = cv2.morphologyEx(imagen_combinada, cv2.MORPH_CLOSE, kernel, iterations=2)
   ```
   
   - Seguido de apertura para eliminar pequeños ruidos:
   ```python
   imagen_combinada = cv2.morphologyEx(imagen_combinada, cv2.MORPH_OPEN, kernel, iterations=1)
   ```

4. **Detección de Contornos**:
   - Encuentra todos los contornos en la imagen preprocesada:
   ```python
   contornos, jerarquia = cv2.findContours(imagen_contornos, metodo, aprox)
   ```

5. **Análisis de Circularidad**:
   - Para cada contorno, calcula área y perímetro:
   ```python
   area = cv2.contourArea(contorno)
   perimetro = cv2.arcLength(contorno, True)
   ```
   
   - Calcula la circularidad como medida de similitud a un círculo perfecto:
   ```python
   circularidad = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
   ```
   
   - Filtra por circularidad mínima (típicamente 0.6 o superior):
   ```python
   if circularidad > circularidad_min:
       # Procesar como círculo válido
   ```

6. **Recuperación de Círculos Parciales**:
   - Guarda contornos rechazados que podrían ser círculos incompletos:
   ```python
   contornos_rechazados.append((contorno, circularidad, area, perimetro))
   ```
   
   - Si se detectan pocos círculos, intenta recuperar círculos parciales con un umbral de circularidad más bajo:
   ```python
   if completar_circulos and len(self.circulos_contornos) < 1:
       # Ordenar por circularidad (de mayor a menor)
       contornos_rechazados.sort(key=lambda x: x[1], reverse=True)
       
       # Intentar recuperar contornos con circularidad algo menor
       for contorno, circularidad, area, perimetro in contornos_rechazados[:5]:
           if circularidad > circularidad_min * 0.7:
               # Procesar como círculo potencial
   ```

7. **Eliminación de Duplicados**:
   - Verifica superposición entre círculos detectados para evitar duplicados:
   ```python
   es_duplicado = False
   for c_centro, c_radio in self.circulos_contornos:
       # Calcular distancia entre centros
       dist = np.sqrt((centro[0] - c_centro[0])**2 + (centro[1] - c_centro[1])**2)
       # Si los centros están cerca y los radios son similares, considerar duplicado
       if dist < max(c_radio, radio) * 0.5 and abs(c_radio - radio) < max(c_radio, radio) * 0.3:
           es_duplicado = True
           break
   ```

8. **Visualización de Resultados**:
   - Dibuja los círculos detectados y sus contornos para depuración:
   ```python
   cv2.circle(self.imagen_con_contornos, centro, radio, (0, 255, 0), 2)
   cv2.circle(self.imagen_con_contornos, centro, 2, (255, 0, 0), 3)
   cv2.drawContours(self.imagen_con_contornos, [contorno], 0, (255, 255, 0), 1)
   ```

**Ventajas del Método Mejorado**:
- **Adaptabilidad**: Funciona bien en diversas condiciones de iluminación gracias al enfoque multi-umbral.
- **Robustez**: Puede detectar círculos parcialmente ocluidos o imperfectos.
- **Eficiencia**: No requiere buscar en un espacio de parámetros 3D como la transformada de Hough.
- **Flexibilidad**: Permite ajustar el umbral de circularidad según la aplicación.

**Desafíos Resueltos**:
- **Círculos Incompletos**: Mediante la recuperación de contornos con circularidad algo menor.
- **Variación de Iluminación**: A través del enfoque multi-umbral y ecualización.
- **Ruido**: Con filtrado previo y operaciones morfológicas.
- **Detecciones Duplicadas**: Mediante verificación de superposición.

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
4. **Ecualización de Histograma**: Se normaliza la distribución de intensidades para mejorar el contraste.
5. **Binarización**: Se convierte la imagen a blanco y negro mediante umbralización.
   - Para Hough: Umbralización simple.
   - Para Contornos: Enfoque multi-umbral que combina resultados de diferentes métodos.
6. **Operaciones Morfológicas**: Se aplican operaciones como cierre o apertura para mejorar la segmentación.
   - El cierre (MORPH_CLOSE) conecta contornos cercanos que podrían ser parte del mismo círculo.
   - La apertura (MORPH_OPEN) elimina pequeños ruidos y suaviza los contornos.

### 4.2 Detección de Círculos

Se implementan dos métodos principales de detección:

#### 4.2.1 Método de Transformada de Hough

1. **Espacio de Parámetros**: La transformada de Hough opera en un espacio tridimensional (centro_x, centro_y, radio).
2. **Proceso de Votación**:
   - Para cada píxel de borde en la imagen, se consideran todos los posibles círculos que podrían pasar por ese punto.
   - Se incrementa un contador en el espacio de Hough para cada combinación (centro_x, centro_y, radio).
   - Los puntos del espacio de Hough con más "votos" corresponden a los círculos más probables.
3. **Configuración Paramétrica**:
   - `dp`: Controla la resolución del acumulador. Si dp=1, tiene la misma resolución que la imagen de entrada.
   - `minDist`: Distancia mínima entre centros de círculos. Ayuda a prevenir múltiples detecciones del mismo círculo.
   - `param1`: Umbral para detector de bordes Canny. Mayor valor implica menos bordes detectados.
   - `param2`: Umbral para aceptar un círculo. Mayor valor implica menos círculos (pero más confiables).
   - `minRadius`/`maxRadius`: Limita el rango de radios a buscar, mejorando la eficiencia y precisión.

#### 4.2.2 Método de Contornos Mejorado

1. **Fundamento Teórico**:
   - La circularidad de una forma se define como: 4π × área / perímetro².
   - Un círculo perfecto tiene circularidad = 1.0.
   - Formas menos circulares tienen valores menores.

2. **Proceso de Detección**:
   - **Umbralización Multi-nivel**: Aplica varios umbrales para crear múltiples imágenes binarias.
   ```python
   # Ejemplo de umbralización multi-nivel
   imagenes_binarias = []
   for umbral in [50, 100, 150, 200]:
       _, img_bin = cv2.threshold(imagen_ecualizada, umbral, 255, cv2.THRESH_BINARY)
       imagenes_binarias.append(img_bin)
   ```
   
   - **Umbralización Adaptativa**: Ajusta el umbral según las condiciones locales de la imagen.
   ```python
   # Umbralización adaptativa
   img_adapt = cv2.adaptiveThreshold(imagen_ecualizada, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
   ```
   
   - **Combinación de Resultados**: Fusiona las múltiples imágenes binarias usando operaciones lógicas.
   ```python
   # Combinación de imágenes binarias
   imagen_combinada = np.zeros_like(imagen)
   for img_bin in imagenes_binarias:
       imagen_combinada = cv2.bitwise_or(imagen_combinada, img_bin)
   ```
   
   - **Mejora Morfológica**: Aplica operaciones morfológicas para mejorar los contornos.
   ```python
   # Operaciones morfológicas
   kernel = np.ones((5, 5), np.uint8)
   imagen_combinada = cv2.morphologyEx(imagen_combinada, cv2.MORPH_CLOSE, kernel, iterations=2)
   imagen_combinada = cv2.morphologyEx(imagen_combinada, cv2.MORPH_OPEN, kernel, iterations=1)
   ```
   
   - **Extracción de Contornos**: Encuentra los contornos en la imagen binaria resultante.
   ```python
   # Detección de contornos
   contornos, _ = cv2.findContours(imagen_contornos, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   ```
   
   - **Análisis de Forma**: Calcula propiedades geométricas y filtra por circularidad.
   ```python
   # Cálculo de circularidad
   area = cv2.contourArea(contorno)
   perimetro = cv2.arcLength(contorno, True)
   circularidad = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
   
   if circularidad > circularidad_min:
       # Procesar como círculo válido
   ```
   
   - **Recuperación de Círculos Parciales**: Intenta recuperar círculos que no cumplen estrictamente con el umbral de circularidad.
   ```python
   # Recuperación de círculos parciales
   if completar_circulos and len(self.circulos_contornos) < 1:
       for contorno, circularidad, area, perimetro in contornos_rechazados[:5]:
           if circularidad > circularidad_min * 0.7:
               # Procesar como círculo potencial
   ```
   
   - **Eliminación de Duplicados**: Evita múltiples detecciones del mismo círculo.
   ```python
   # Verificación de duplicados
   es_duplicado = False
   for c_centro, c_radio in self.circulos_contornos:
       dist = np.sqrt((centro[0] - c_centro[0])**2 + (centro[1] - c_centro[1])**2)
       if dist < max(c_radio, radio) * 0.5:
           es_duplicado = True
           break
   ```

### 4.3 Análisis y Medición

Una vez detectados los círculos, se realizan las siguientes mediciones:

1. **Radio**: Se extrae directamente de la detección o se calcula a partir del contorno.
   ```python
   # Para el método de Hough
   radio = circulo[2]  # El tercer elemento es el radio
   
   # Para el método de contornos
   (x, y), radio = cv2.minEnclosingCircle(contorno)
   ```

2. **Área**: Se calcula como π × r² o directamente del contorno.
   ```python
   # Cálculo basado en radio
   area = np.pi * radio ** 2
   
   # Cálculo directo del contorno
   area = cv2.contourArea(contorno)
   ```

3. **Perímetro**: Se calcula como 2π × r o directamente del contorno.
   ```python
   # Cálculo basado en radio
   perimetro = 2 * np.pi * radio
   
   # Cálculo directo del contorno
   perimetro = cv2.arcLength(contorno, True)
   ```

4. **Distribución Estadística**: Se calculan estadísticas como media, mínimo, máximo y desviación estándar.
   ```python
   # Estadísticas de los círculos detectados
   radio_medio = np.mean(self.radios)
   radio_max = np.max(self.radios)
   radio_min = np.min(self.radios)
   radio_std = np.std(self.radios)
   ```

### 4.4 Generación de Resultados

Los resultados se presentan en múltiples formatos:

1. **Visualización en Pantalla**: Imágenes con círculos marcados y estadísticas básicas.
2. **Archivos Excel**: Tablas detalladas con todas las métricas calculadas.
3. **Informes PDF**: Documentos estructurados con análisis, gráficos y conclusiones.
4. **Documentación HTML**: Análisis comparativo con visualizaciones interactivas.

## 5. Evaluación y Comparación de Métodos

### 5.1 Comparativa de Métodos de Detección

| Aspecto | Transformada de Hough | Método de Contornos Mejorado |
|---------|----------------------|---------------------|
| Precisión | Alta para círculos perfectos | Alta con el enfoque multi-umbral |
| Velocidad | Más lenta (especialmente para rangos amplios de radio) | Más rápida (especialmente con filtrado de área) |
| Robustez ante ruido | Media (sensible a param1/param2) | Alta con el preprocesamiento mejorado |
| Detección de círculos parciales | Buena (depende de param2) | Mejorada con recuperación adaptativa |
| Personalización | Muchos parámetros para ajustar | Flexible con umbral de circularidad adaptable |
| Complejidad computacional | Alta (O(n³) en el peor caso) | Media (O(n²) en el peor caso) |
| Consumo de memoria | Alto para imágenes grandes | Moderado |

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
   - El método de contornos mejorado con enfoque multi-umbral proporciona excelentes resultados con mayor eficiencia computacional.

2. **Importancia del Preprocesamiento**:
   - La aplicación de filtros (especialmente gaussiano) mejora significativamente la detección.
   - Las operaciones morfológicas son cruciales para obtener buenos contornos, especialmente el cierre para rellenar discontinuidades.
   - La ecualización de histograma normaliza las imágenes con iluminación variable.

3. **Efecto de los Parámetros**:
   - La selección adecuada de parámetros (umbral, tamaño de kernel, etc.) es crítica y depende de las características específicas de la imagen.
   - El enfoque multi-umbral reduce la dependencia de un único valor óptimo de umbral.

### 6.2 Recomendaciones Prácticas

1. **Para Imágenes Ruidosas**:
   - Aplicar filtrado gaussiano con kernel 5x5 o mayor
   - Usar operaciones morfológicas de cierre con iteraciones múltiples
   - Utilizar el método de contornos mejorado con umbral de circularidad reducido (0.5-0.6)

2. **Para Procesamiento en Tiempo Real**:
   - Redimensionar imágenes a tamaño medio (400x400)
   - Utilizar método de contornos con área mínima mayor
   - Limitar el número de niveles en el enfoque multi-umbral

3. **Para Máxima Precisión**:
   - Usar formatos sin pérdida (PNG, BMP)
   - Aplicar filtros de nitidez después del suavizado
   - Combinar ambos métodos de detección y comparar resultados
   - Utilizar el enfoque multi-umbral completo

4. **Para Detección Robusta**:
   - Implementar un enfoque en dos fases: detección aproximada seguida de refinamiento
   - Utilizar la recuperación de círculos parciales
   - Experimentar con diferentes valores de circularidad según la aplicación

## 7. Referencias

1. OpenCV Documentation: https://docs.opencv.org/
2. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
3. Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
4. Bradski, G., & Kaehler, A. (2008). Learning OpenCV. O'Reilly Media.
5. Davies, E. R. (2017). Computer Vision: Principles, Algorithms, Applications, Learning. Academic Press.
