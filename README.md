# Detección de Señales de Tráfico por Color usando OpenCV

Este proyecto utiliza **OpenCV** y **NumPy** para detectar señales de tráfico en una imagen basándose en sus colores. El programa lee una imagen, convierte el espacio de color a **HSV**, aplica filtros por rango de color y dibuja contornos alrededor de las señales detectadas.

---

## 🛠️ **Descripción del proceso**
### 1. **Lectura y conversión de la imagen**
- Se carga la imagen cambiando el contenido de la variable (`source`) en formato BGR usando `cv.imread`.
- La imagen se convierte al espacio de color **HSV** utilizando `cv.cvtColor`, ya que HSV facilita la segmentación por color.

---

## 🧪 **Orden de los filtros aplicados**
Los filtros y transformaciones se aplican en el siguiente orden para cada color:

1. **Creación de máscaras por rango de color**  
   - Se definen dos rangos para cada color para mejorar la precisión en la detección.

2. **Unión de las máscaras**  
   - Se combinan las dos máscaras para cada color utilizando `cv.bitwise_or`.

3. **Erosión para reducir el ruido**  
   - Se aplica `cv.erode` para eliminar el ruido y mejorar la calidad de la máscara.

4. **Aplicación de la máscara**  
   - Se aplica `cv.bitwise_and` para extraer las áreas del color correspondiente en la imagen original.

5. **Dibujar contornos**  
   - Se detectan y dibujan los contornos con `cv.findContours` y `cv.rectangle`.

6. **Clasificación de señales**  
   - Se evalúa el área detectada para determinar la clasificación de la señal.

---

## 🎯 **Definición de rangos de color**
Se definen dos rangos para cada color para mejorar la precisión de la detección:

- **Rojo**:
  - Primer rango: `[0, 100, 100]` a `[10, 255, 255]`
  - Segundo rango: `[160, 100, 100]` a `[180, 255, 255]`

- **Verde**:
  - Primer rango: `[35, 100, 100]` a `[85, 255, 255]`
  - Segundo rango: `[85, 50, 100]` a `[95, 255, 255]`

- **Azul**:
  - Primer rango: `[85, 100, 100]` a `[130, 255, 255]`
  - Segundo rango: `[130, 50, 100]` a `[140, 255, 255]`

- **Amarillo**:
  - Primer rango: `[20, 100, 100]` a `[35, 255, 255]`
  - Segundo rango: `[35, 50, 100]` a `[45, 255, 255]`

---

## 🚦 **Clasificación de señales**
Para que un color se clasifique como una señal válida, el área detectada debe superar un **umbral de 5000 píxeles** (`threshold_area`):

```python
threshold_area = 5000  # Número mínimo de píxeles para considerar la presencia de color
