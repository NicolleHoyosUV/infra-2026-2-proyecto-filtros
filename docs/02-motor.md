# El motor: filtros, paralelización y mediciones

## 1. Convenciones generales y corrección

Todos los filtros implementados en `motor/src/filtros.cpp` respetan las convenciones exigidas para garantizar exactitud numérica y equivalencia byte a byte entre las versiones secuencial y paralela:

1. **Tratamiento de bordes:** Se aplica repetición del píxel más cercano mediante la función `recortar(v, limite)`, restringiendo las coordenadas de lectura al rango válido $[0, W-1]$ en horizontal y $[0, H-1]$ en vertical sin rellenar con ceros ni recortar dimensiones.
2. **Aritmética en doble precisión:** Todas las acumulaciones y pesos de convolución se calculan en coma flotante `double`. La conversión a entero de 8 bits (`unsigned char`) se realiza únicamente al escribir el píxel final mediante `redondear(v)` ($\lfloor v + 0.5 \rfloor$ saturado al intervalo $[0, 255]$).
3. **Verificación byte a byte:** El script automatizado `motor/tests/pruebas.sh` ejecuta el comparador binario (`./comparar`) sobre imágenes de prueba y sobre una imagen sintética de $512 \times 512$ verificando que la salida paralela con $1$ y $8$ hilos sea idéntica byte a byte a la versión secuencial tanto para cada filtro individual (`gaussiano`, `sobel`, `otsu`) como para la cadena completa (`gris,gaussiano,sobel,otsu`). El motor compila limpio sin advertencias bajo `-std=c++17 -O2 -Wall -Wextra -fopenmp`.

---

## 2. Implementación de los filtros y estrategia de paralelización

### 2.1. Desenfoque gaussiano separable

* **Implementación:** El filtro suaviza la imagen utilizando una máscara de $5 \times 5$ factorizada en dos pasadas unidimensionales con el núcleo binomial:
  $$k = \frac{1}{16} \begin{bmatrix} 1 & 4 & 6 & 4 & 1 \end{bmatrix}$$
  En la primera pasada (horizontal) se calcula $T(x, y) = \sum_{i=-2}^{2} k_{i} \, I(x+i, y)$ almacenando los resultados en un arreglo intermedio `std::vector<double>` para evitar pérdida de precisión por redondeo intermedio. En la segunda pasada (vertical) se evalúa $O(x, y) = \text{redondeo}\left(\sum_{j=-2}^{2} k_{j} \, T(x, y+j)\right)$ sobre dicho arreglo intermedio. Adicionalmente, se implementó la variante directa 2D (`gaussiano_directo`, $25$ multiplicaciones por píxel frente a $10$ de la separable) para contrastar la ganancia algorítmica.
* **Estrategia OpenMP:** Patrón *stencil* en dos etapas. Ambas pasadas se paralelizan con `#pragma omp parallel for schedule(static)` sobre el bucle externo de las filas (`y`), existiendo una barrera implícita de sincronización entre la pasada horizontal y la vertical para garantizar que `temp` esté completamente calculado antes de ser leído verticalmente.

### 2.2. Sobel y magnitud del gradiente

* **Implementación:** Opera sobre imágenes de un solo canal (PGM) estimando las derivadas espaciales mediante las máscaras $G_x$ y $G_y$ de $3 \times 3$. Para maximizar el aprovechamiento de caché, ambos gradientes ($g_x$ y $g_y$) se calculan simultáneamente en una **única lectura de la vecindad** de $3 \times 3$, reduciendo a la mitad los accesos a memoria frente a dos pasadas separadas. La salida final aplica saturación a $255$:
  $$O(x, y) = \min\left(255, \text{redondeo}\left(\sqrt{g_x^2 + g_y^2}\right)\right)$$
* **Estrategia OpenMP:** Patrón *stencil* sin dependencias entre píxeles de salida. Se reparte el bucle externo de filas con `#pragma omp parallel for schedule(static)`.

### 2.3. Histograma y umbral de Otsu

* **Implementación:** El filtro consta de tres etapas:
  1. Construcción del histograma $h(v)$ para $v \in [0, 255]$ sobre enteros (`std::size_t`), garantizando determinismo independiente del orden de ejecución de los hilos.
  2. Búsqueda del umbral óptimo $t^* \in [0, 254]$ que maximiza la varianza entre clases:
     $$\sigma_{B}^{2}(t) = \omega_{0}(t)\omega_{1}(t)\left(\mu_{0}(t) - \mu_{1}(t)\right)^{2}$$
  3. Binarización de la imagen asignando $O(x, y) = 0$ si $I(x, y) \le t^*$ y $255$ en caso contrario.
* **Estrategia OpenMP y *false sharing*:**
  * **Histograma (Patrón reducción):** Actualizar un único arreglo compartido con `#pragma omp atomic` (`otsu_compartido`) serializa los accesos a memoria y degrada el rendimiento por contención. Para evitarlo, `otsu_paralelo` asigna un histograma privado por hilo dentro de una estructura alineada a línea de caché (`struct alignas(64) HistogramaAlineado`), eliminando el *false sharing* entre hilos contiguos en memoria. Al finalizar el recorrido paralelo, se realiza una reducción secuencial de los $p$ histogramas privados ($p \times 256$ sumas).
  * **Barrido de umbrales (Ley de Amdahl):** El cálculo de $\sigma_B^2(t)$ sobre los $255$ niveles posibles se mantiene secuencial, ya que su costo computacional frente a los $N = W \cdot H$ píxeles de la imagen es despreciable ($255 \ll 16{,}777{,}216$ en una imagen de $4096 \times 4096$).
  * **Binarización (Patrón *map*):** Se reparte por filas con `#pragma omp parallel for schedule(static)`.

### 2.4. Justificación del reparto por filas y planificación estática

En todos los filtros paralelos se eligió repartir el ciclo externo de las **filas (`y`)** y no el de las columnas (`x`) debido a la organización de la memoria en C++ (*row-major order*): los píxeles contiguos de una misma fila ocupan direcciones adyacentes en RAM. Al asignar bandas horizontales contiguas a cada hilo, las lecturas secuenciales cargan líneas de caché completas útiles (*spatial locality*), mientras que repartir por columnas provocaría saltos de $W \times \text{canales}$ bytes entre iteraciones consecutivas y invalidaciones constantes de caché. Se empleó `schedule(static)` porque la carga de trabajo por fila es uniforme y predecible, evitando la sobrecarga de planificación dinámica en tiempo de ejecución.

---

## 3. Entorno de medición y resultados experimentales

*(Sección reservada para las mediciones sobre la imagen de $4096 \times 4096$)*

### 3.1. Especificaciones de la máquina de pruebas

* **Procesador:** `[COMPLETAR: Modelo de CPU, núcleos físicos e hilos]`
* **Memoria RAM:** `[COMPLETAR: Capacidad y tipo]`
* **Compilador y SO:** `[COMPLETAR: Versión de g++ y distribución Linux]`

### 3.2. Comparaciones algorítmicas y de contención

* **Gaussiano separable vs. directo $5 \times 5$ (1 hilo):** `[COMPLETAR: Tiempos de gaussiano vs gaussiano_directo]`
* **Histograma privado alineado vs. compartido (`atomic`):** `[COMPLETAR: Tiempos de otsu vs otsu_compartido]`

### 3.3. Tiempos, Speedup y Eficiencia ($p \in \{1, 2, 4, 8\}$)

Con base en los datos registrados en `mediciones/motor.csv` (mediana de 4 ejecuciones tras descartar la primera corrida de calentamiento), se calculan el speedup $S(p) = \frac{T(1)}{T(p)}$ y la eficiencia $E(p) = \frac{S(p)}{p}$:

| Filtro | $T(1)$ [ms] | $T(2)$ [ms] | $S(2)$ | $E(2)$ | $T(4)$ [ms] | $S(4)$ | $E(4)$ | $T(8)$ [ms] | $S(8)$ | $E(8)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gaussiano` | - | - | - | - | - | - | - | - | - | - |
| `sobel` | - | - | - | - | - | - | - | - | - | - |
| `otsu` | - | - | - | - | - | - | - | - | - | - |
| Cadena completa | - | - | - | - | - | - | - | - | - | - |

### 3.4. Perfilado y optimización de localidad de caché

* **Evidencia del perfilador:** `[COMPLETAR: Salida de perf stat o valgrind --tool=callgrind y análisis]`
* **Optimización de localidad aplicada (antes y después):** `[COMPLETAR: Medición y explicación del cambio en acceso a memoria]`

