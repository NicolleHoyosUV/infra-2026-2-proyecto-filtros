#include "filtros.hpp"

#include <omp.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <stdexcept>

Imagen gris_secuencial(const Imagen& e) {
    if (e.canales != 3) throw std::runtime_error("gris espera una imagen de tres canales");
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(static_cast<std::size_t>(e.ancho) * e.alto);
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            const double v = 0.299 * e.en(x, y, 0) + 0.587 * e.en(x, y, 1) + 0.114 * e.en(x, y, 2);
            s.poner(x, y, 0, redondear(v));
        }
    }
    return s;
}

Imagen gris_paralelo(const Imagen& e) {
    if (e.canales != 3) throw std::runtime_error("gris espera una imagen de tres canales");
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(static_cast<std::size_t>(e.ancho) * e.alto);
    // Cada fila es independiente de las demás: no hay nada que compartir
    // entre hilos ni dependencias entre iteraciones.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            const double v = 0.299 * e.en(x, y, 0) + 0.587 * e.en(x, y, 1) + 0.114 * e.en(x, y, 2);
            s.poner(x, y, 0, redondear(v));
        }
    }
    return s;
}

namespace {

// El promedio de la vecindad de 3x3 de un canal, con las coordenadas
// recortadas al rectángulo de la imagen.
double promedio_vecindad(const Imagen& e, int x, int y, int c) {
    double suma = 0.0;
    for (int j = -1; j <= 1; ++j) {
        const int fila = recortar(y + j, e.alto);
        for (int i = -1; i <= 1; ++i) {
            suma += e.en(recortar(x + i, e.ancho), fila, c);
        }
    }
    return suma / 9.0;
}

// Núcleo binomial 1D: [1 4 6 4 1] / 16
constexpr double KERNEL_GAUSS[5] = {1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0};

// Estructura alineada a 64 bytes (tamaño típico de línea de caché) para
// evitar false sharing entre hilos al acumular histogramas privados.
struct alignas(64) HistogramaAlineado {
    std::array<std::size_t, 256> cuentas{};
};

// Busca el umbral óptimo de Otsu sobre un histograma de 256 niveles.
int calcular_umbral_otsu(const std::array<std::size_t, 256>& hist, std::size_t total_pixeles) {
    double suma_total = 0.0;
    for (int v = 0; v < 256; ++v) {
        suma_total += static_cast<double>(v) * static_cast<double>(hist[v]);
    }

    std::size_t peso_fondo = 0;
    double suma_fondo = 0.0;
    double max_varianza = -1.0;
    int mejor_umbral = 0;

    for (int t = 0; t < 255; ++t) {
        peso_fondo += hist[t];
        if (peso_fondo == 0) continue;

        const std::size_t peso_objeto = total_pixeles - peso_fondo;
        if (peso_objeto == 0) break;

        suma_fondo += static_cast<double>(t) * static_cast<double>(hist[t]);

        const double w0 = static_cast<double>(peso_fondo) / static_cast<double>(total_pixeles);
        const double w1 = static_cast<double>(peso_objeto) / static_cast<double>(total_pixeles);
        const double mu0 = suma_fondo / static_cast<double>(peso_fondo);
        const double mu1 = (suma_total - suma_fondo) / static_cast<double>(peso_objeto);

        const double diff = mu0 - mu1;
        const double varianza = w0 * w1 * diff * diff;

        if (varianza > max_varianza) {
            max_varianza = varianza;
            mejor_umbral = t;
        }
    }
    return mejor_umbral;
}

}  // namespace

Imagen caja_secuencial(const Imagen& e) {
    Imagen s = e;
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                s.poner(x, y, c, redondear(promedio_vecindad(e, x, y, c)));
            }
        }
    }
    return s;
}

Imagen caja_paralelo(const Imagen& e) {
    Imagen s = e;
    // El ciclo externo recorre filas y el interno columnas, que es el orden
    // en que la imagen está en memoria. Repartir por filas deja a cada hilo
    // con una banda contigua.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                s.poner(x, y, c, redondear(promedio_vecindad(e, x, y, c)));
            }
        }
    }
    return s;
}

Imagen gaussiano_secuencial(const Imagen& e) {
    const std::size_t total = static_cast<std::size_t>(e.ancho) * e.alto * e.canales;
    std::vector<double> temp(total, 0.0);
    Imagen s = e;

    // Pasada 1: Horizontal (mantiene precisión en double sin redondeo intermedio)
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                double suma = 0.0;
                for (int i = -2; i <= 2; ++i) {
                    const int col = recortar(x + i, e.ancho);
                    suma += KERNEL_GAUSS[i + 2] * e.en(col, y, c);
                }
                temp[(static_cast<std::size_t>(y) * e.ancho + x) * e.canales + c] = suma;
            }
        }
    }
    // Pasada 2: Vertical sobre el buffer intermedio y redondeo final
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                double suma = 0.0;
                for (int j = -2; j <= 2; ++j) {
                    const int fila = recortar(y + j, e.alto);
                    suma += KERNEL_GAUSS[j + 2] *
                            temp[(static_cast<std::size_t>(fila) * e.ancho + x) * e.canales + c];
                }
                s.poner(x, y, c, redondear(suma));
            }
        }
    }

    return s;
}

Imagen gaussiano_paralelo(const Imagen& e) {
    const std::size_t total = static_cast<std::size_t>(e.ancho) * e.alto * e.canales;
    std::vector<double> temp(total, 0.0);
    Imagen s = e;

    // Pasada 1: Horizontal repartida por filas
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                double suma = 0.0;
                for (int i = -2; i <= 2; ++i) {
                    const int col = recortar(x + i, e.ancho);
                    suma += KERNEL_GAUSS[i + 2] * e.en(col, y, c);
                }
                temp[(static_cast<std::size_t>(y) * e.ancho + x) * e.canales + c] = suma;
            }
        }
    }

    // Pasada 2: Vertical repartida por filas (barrera implícita tras la pasada 1)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                double suma = 0.0;
                for (int j = -2; j <= 2; ++j) {
                    const int fila = recortar(y + j, e.alto);
                    suma += KERNEL_GAUSS[j + 2] *
                            temp[(static_cast<std::size_t>(fila) * e.ancho + x) * e.canales + c];
                }
                s.poner(x, y, c, redondear(suma));
            }
        }
    }

    return s;
}

Imagen sobel_secuencial(const Imagen& e) {
    if (e.canales != 1) throw std::runtime_error("sobel espera una imagen de un canal");
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(static_cast<std::size_t>(e.ancho) * e.alto);

    for (int y = 0; y < e.alto; ++y) {
        const int y0 = recortar(y - 1, e.alto);
        const int y1 = y;
        const int y2 = recortar(y + 1, e.alto);

        for (int x = 0; x < e.ancho; ++x) {
            const int x0 = recortar(x - 1, e.ancho);
            const int x1 = x;
            const int x2 = recortar(x + 1, e.ancho);

            const double p00 = e.en(x0, y0, 0);
            const double p01 = e.en(x1, y0, 0);
            const double p02 = e.en(x2, y0, 0);
            const double p10 = e.en(x0, y1, 0);
            const double p12 = e.en(x2, y1, 0);
            const double p20 = e.en(x0, y2, 0);
            const double p21 = e.en(x1, y2, 0);
            const double p22 = e.en(x2, y2, 0);

            // Ambas convoluciones en una sola lectura de la vecindad
            const double gx = (-p00 + p02) + (-2.0 * p10 + 2.0 * p12) + (-p20 + p22);
            const double gy = (-p00 - 2.0 * p01 - p02) + (p20 + 2.0 * p21 + p22);

            const double magnitud = std::sqrt(gx * gx + gy * gy);
            s.poner(x, y, 0, redondear(magnitud));
        }
    }
    return s;
}

Imagen sobel_paralelo(const Imagen& e) {
    if (e.canales != 1) throw std::runtime_error("sobel espera una imagen de un canal");
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(static_cast<std::size_t>(e.ancho) * e.alto);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        const int y0 = recortar(y - 1, e.alto);
        const int y1 = y;
        const int y2 = recortar(y + 1, e.alto);

        for (int x = 0; x < e.ancho; ++x) {
            const int x0 = recortar(x - 1, e.ancho);
            const int x1 = x;
            const int x2 = recortar(x + 1, e.ancho);

            const double p00 = e.en(x0, y0, 0);
            const double p01 = e.en(x1, y0, 0);
            const double p02 = e.en(x2, y0, 0);
            const double p10 = e.en(x0, y1, 0);
            const double p12 = e.en(x2, y1, 0);
            const double p20 = e.en(x0, y2, 0);
            const double p21 = e.en(x1, y2, 0);
            const double p22 = e.en(x2, y2, 0);

            const double gx = (-p00 + p02) + (-2.0 * p10 + 2.0 * p12) + (-p20 + p22);
            const double gy = (-p00 - 2.0 * p01 - p02) + (p20 + 2.0 * p21 + p22);

            const double magnitud = std::sqrt(gx * gx + gy * gy);
            s.poner(x, y, 0, redondear(magnitud));
        }
    }
    return s;
}

Imagen otsu_secuencial(const Imagen& e) {
    if (e.canales != 1) throw std::runtime_error("otsu espera una imagen de un canal");
    const std::size_t total_pixeles = static_cast<std::size_t>(e.ancho) * e.alto;

    // 1. Construcción del histograma
    std::array<std::size_t, 256> hist{};
    for (std::size_t i = 0; i < total_pixeles; ++i) {
        hist[e.datos[i]]++;
    }

    // 2. Búsqueda del umbral óptimo (255 iteraciones)
    const int umbral = calcular_umbral_otsu(hist, total_pixeles);

    // 3. Binarización de la imagen
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(total_pixeles);

    for (std::size_t i = 0; i < total_pixeles; ++i) {
        s.datos[i] = (e.datos[i] <= umbral) ? 0 : 255;
    }

    return s;
}

Imagen otsu_paralelo(const Imagen& e) {
    if (e.canales != 1) throw std::runtime_error("otsu espera una imagen de un canal");
    const std::size_t total_pixeles = static_cast<std::size_t>(e.ancho) * e.alto;

    // 1. Histograma paralelo con acumuladores privados alineados a línea de caché
    const int max_hilos = omp_get_max_threads();
    std::vector<HistogramaAlineado> privados(max_hilos);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& hist_local = privados[tid].cuentas;

        #pragma omp for schedule(static)
        for (int y = 0; y < e.alto; ++y) {
            const std::size_t fila_offset = static_cast<std::size_t>(y) * e.ancho;
            for (int x = 0; x < e.ancho; ++x) {
                hist_local[e.datos[fila_offset + x]]++;
            }
        }
    }

    // Reducción final de los histogramas privados
    std::array<std::size_t, 256> hist{};
    for (int t = 0; t < max_hilos; ++t) {
        for (int v = 0; v < 256; ++v) {
            hist[v] += privados[t].cuentas[v];
        }
    }

    // 2. Barrido de 255 umbrales (secuencial por Ley de Amdahl)
    const int umbral = calcular_umbral_otsu(hist, total_pixeles);

    // 3. Binarización paralela por filas
    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(total_pixeles);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        const std::size_t fila_offset = static_cast<std::size_t>(y) * e.ancho;
        for (int x = 0; x < e.ancho; ++x) {
            const std::size_t idx = fila_offset + x;
            s.datos[idx] = (e.datos[idx] <= umbral) ? 0 : 255;
        }
    }

    return s;
}

// ---------------------------------------------------------------------------
// Variantes experimentales para las comparaciones del informe
// ---------------------------------------------------------------------------

// Gaussiano directo 5x5 en una sola pasada (25 multiplicaciones por píxel).
// Permite demostrar la mejora algorítmica del separable (10 multiplicaciones).
Imagen gaussiano_directo(const Imagen& e) {
    static constexpr double KERNEL_2D[5][5] = {
        {1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0},
        {4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0},
        {6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0},
        {4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0},
        {1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0}
    };

    Imagen s = e;
    for (int y = 0; y < e.alto; ++y) {
        for (int x = 0; x < e.ancho; ++x) {
            for (int c = 0; c < e.canales; ++c) {
                double suma = 0.0;
                for (int j = -2; j <= 2; ++j) {
                    const int fila = recortar(y + j, e.alto);
                    for (int i = -2; i <= 2; ++i) {
                        const int col = recortar(x + i, e.ancho);
                        suma += KERNEL_2D[j + 2][i + 2] * e.en(col, fila, c);
                    }
                }
                s.poner(x, y, c, redondear(suma));
            }
        }
    }
    return s;
}

// Otsu paralelo ingenuo con acumulador compartido protegido por atomic.
// Permite medir la contención frente a la versión con histogramas privados.
Imagen otsu_compartido(const Imagen& e) {
    if (e.canales != 1) throw std::runtime_error("otsu espera una imagen de un canal");
    const std::size_t total_pixeles = static_cast<std::size_t>(e.ancho) * e.alto;

    std::size_t hist_raw[256] = {0};

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        const std::size_t fila_offset = static_cast<std::size_t>(y) * e.ancho;
        for (int x = 0; x < e.ancho; ++x) {
            const unsigned char val = e.datos[fila_offset + x];
            #pragma omp atomic
            hist_raw[val]++;
        }
    }

    std::array<std::size_t, 256> hist{};
    for (int v = 0; v < 256; ++v) {
        hist[v] = hist_raw[v];
    }

    const int umbral = calcular_umbral_otsu(hist, total_pixeles);

    Imagen s;
    s.ancho = e.ancho;
    s.alto = e.alto;
    s.canales = 1;
    s.datos.resize(total_pixeles);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < e.alto; ++y) {
        const std::size_t fila_offset = static_cast<std::size_t>(y) * e.ancho;
        for (int x = 0; x < e.ancho; ++x) {
            const std::size_t idx = fila_offset + x;
            s.datos[idx] = (e.datos[idx] <= umbral) ? 0 : 255;
        }
    }

    return s;
}
