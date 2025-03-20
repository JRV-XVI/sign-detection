# Detección de Señales de Tráfico por Color usando OpenCV

Este proyecto utiliza **OpenCV** y **NumPy** para detectar señales de tráfico en una imagen basándose en sus colores. El programa lee una imagen, convierte el espacio de color a **HSV**, aplica filtros por rango de color y clasifica las señales detectadas según su color predominante.

## 📋 Requisitos

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## 🚀 Uso

1. Coloque sus imágenes en el directorio `./images/`
2. Modifique la variable `source` en el código para apuntar a la imagen deseada
3. Ejecute el script:
   ```
   python main.py
   ```

## 🛠️ Descripción del proceso

### 1. Lectura y conversión de la imagen
- Se carga la imagen desde la ruta especificada en la variable `source` en formato BGR usando `cv.imread`
- La imagen se convierte al espacio de color **HSV** utilizando `cv.cvtColor`, ya que HSV facilita la segmentación por color

### 2. Definición de rangos de color
Se definen rangos específicos para cada color en el espacio HSV:
- **Rojo**:
  - Rango inferior: `[0, 120, 100]` a `[10, 255, 255]`
  - Rango superior: `[160, 120, 100]` a `[180, 255, 255]`
- **Verde**: `[40, 120, 100]` a `[80, 255, 255]`
- **Azul**: `[100, 150, 100]` a `[130, 255, 255]`
- **Amarillo**: `[20, 120, 100]` a `[35, 255, 255]`

### 3. Procesamiento y análisis de la imagen

#### 3.1 Creación de máscaras por color
- Se crean máscaras para cada color utilizando `cv.inRange`
- Para el rojo, se combinan dos máscaras para cubrir ambos extremos del espectro HSV

#### 3.2 Operaciones morfológicas
- **Erosión**: Se aplica `cv.erode` con un kernel de 5x5 para eliminar el ruido pequeño
- **Dilatación**: Se aplica `cv.dilate` para restaurar el tamaño de los objetos principales

#### 3.3 Extracción de regiones de color
- Se aplica `cv.bitwise_and` para extraer las áreas del color correspondiente en la imagen original

#### 3.4 Análisis de contornos y clasificación
- Se implementa la función `check_color_presence()` que:
  - Detecta contornos en la máscara usando `cv.findContours`
  - Analiza cada contorno por área y proporción de aspecto
  - Considera válidos los contornos con área > 5000 píxeles y proporción de aspecto entre 0.5 y 2.0

### 4. Clasificación de señales de tráfico
El programa clasifica las señales en cuatro categorías según su color predominante:
- **Rojo**: Señales restrictivas
- **Amarillo**: Señales preventivas
- **Azul**: Señales de servicios
- **Verde**: Señales de destino

### 5. Visualización de resultados
- Se muestra la imagen original junto con las máscaras de cada color
- Se imprime en consola los tipos de señales detectadas

## 🧪 Orden de los filtros aplicados
1. **Creación de máscaras por rango de color**
2. **Aplicación de las máscaras a la imagen original**
3. **Erosión para reducir el ruido**
4. **Dilatación para restaurar objetos principales**
5. **Detección de contornos y clasificación**

## ⚠️ Restricciones y limitaciones
- El programa **solo utiliza el color** para detectar señales, lo que puede generar falsos positivos si el fondo u otros objetos tienen colores similares a las señales de tráfico y además una proporción de aspecto similar a esta
- La iluminación y las condiciones ambientales pueden afectar significativamente la detección basada en color
- El umbral de área (5000 píxeles) podría necesitar ajustes según el tamaño de la imagen y la distancia a las señales

## 🔄 Posibles mejoras
- Implementar detección de forma además del color
- Añadir clasificación mediante aprendizaje automático
- Mejorar la robustez frente a diferentes condiciones de iluminación
- Optimizar los rangos de color para reducir falsos positivos
