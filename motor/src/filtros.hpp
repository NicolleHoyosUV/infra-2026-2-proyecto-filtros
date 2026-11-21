#ifndef FILTROS_HPP
#define FILTROS_HPP

#include "imagen.hpp"

// Escala de grises con la ponderación BT.601. Entra un PPM de color y sale
// un PGM de un canal. Cada píxel de salida depende solo del píxel de entrada
// en la misma posición.
Imagen gris_secuencial(const Imagen& e);
Imagen gris_paralelo(const Imagen& e);

// Desenfoque de caja: cada píxel pasa a ser el promedio de los nueve de su
// vecindad de 3x3. Trabaja igual sobre uno o tres canales.
Imagen caja_secuencial(const Imagen& e);
Imagen caja_paralelo(const Imagen& e);

// Desenfoque gaussiano separable con el núcleo binomial [1 4 6 4 1]/16, en
// una pasada horizontal y otra vertical sobre el resultado intermedio.
Imagen gaussiano_secuencial(const Imagen& e);
Imagen gaussiano_paralelo(const Imagen& e);

// Magnitud del gradiente de Sobel sobre una imagen de un canal.
Imagen sobel_secuencial(const Imagen& e);
Imagen sobel_paralelo(const Imagen& e);

// Umbral de Otsu: busca el umbral que maximiza la varianza entre clases del
// histograma y binariza con él.
Imagen otsu_secuencial(const Imagen& e);
Imagen otsu_paralelo(const Imagen& e);

// Variantes experimentales para instrumentación y comparación en el informe:
// 1) Gaussiano directo 2D (5x5 = 25 multiplicaciones por píxel)
Imagen gaussiano_directo(const Imagen& e);
// 2) Otsu con histograma compartido protegido con #pragma omp atomic
Imagen otsu_compartido(const Imagen& e);

#endif
