#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <omp.h> 

// Estructura para representar un píxel RGB
struct Pixel {
    uint8_t r, g, b;
};

// Constantes de la imagen (Resolución 8K)
const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 1000;

// TAREA A: Generación del Conjunto de Mandelbrot (Paralelizado)
std::vector<Pixel> generateMandelbrot(int width, int height) {
    std::vector<Pixel> image(width * height);
    
    double min_re = -2.5, max_re = 1.0;
    double min_im = -1.0, max_im = 1.0;
    
    double re_factor = (max_re - min_re) / (width - 1);
    double im_factor = (max_im - min_im) / (height - 1);

    // Paralelización con OpenMP usando planificación dinámica por la carga irregular
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        double c_im = max_im - y * im_factor;
        for (int x = 0; x < width; ++x) {
            double c_re = min_re + x * re_factor;
            
            double Z_re = c_re, Z_im = c_im;
            bool isInside = true;
            int n;
            
            for (n = 0; n < MAX_ITER; ++n) {
                double Z_re2 = Z_re * Z_re, Z_im2 = Z_im * Z_im;
                if (Z_re2 + Z_im2 > 4.0) {
                    isInside = false;
                    break;
                }
                Z_im = 2.0 * Z_re * Z_im + c_im;
                Z_re = Z_re2 - Z_im2 + c_re;
            }
            
            int idx = y * width + x;
            if (isInside) {
                image[idx] = {0, 0, 0};
            } else {
                uint8_t r = (n * 9) % 256;
                uint8_t g = (n * 5) % 256;
                uint8_t b = (n * 13) % 256;
                image[idx] = {r, g, b};
            }
        }
    }
    return image;
}

// TAREA B: Filtro de Convolución 2D Pesado (SPMD / SIMD Vectorizado)
std::vector<Pixel> applyGaussianBlur(const std::vector<Pixel>& input, int width, int height, int radius) {
    std::vector<Pixel> output(width * height);
    int size = 2 * radius + 1;
    
    // 1. Generar el Kernel Gaussiano dinámicamente
    std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
    double sum = 0.0;
    double sigma = std::max(radius / 2.0, 1.0);
    
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            double exponent = -(x * x + y * y) / (2.0 * sigma * sigma);
            double val = std::exp(exponent);
            kernel[y + radius][x + radius] = val;
            sum += val;
        }
    }
    
    // Normalizar el kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    // 2. APLICAR EL FILTRO 
    // Paralelización con OpenMP a nivel de hilo (Thread level SPMD)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double r = 0.0, g = 0.0, b = 0.0;
            
            for (int ky = -radius; ky <= radius; ++ky) {
                // Evaluamos el borde Y fuera del bucle interno para evitar cálculos redundantes
                int img_y = std::min(std::max(y + ky, 0), height - 1);
                int row_offset = img_y * width;
                int kernel_y_idx = ky + radius;
                
                // Forzamos la vectorización (SIMD level SPMD) e indicamos una reducción
                #pragma omp simd reduction(+:r, g, b)
                for (int kx = -radius; kx <= radius; ++kx) {
                    int img_x = std::min(std::max(x + kx, 0), width - 1);
                    int pixel_idx = row_offset + img_x;
                    
                    double weight = kernel[kernel_y_idx][kx + radius];
                    
                    // Estas multiplicaciones y sumas se harán en paralelo en los registros vectoriales
                    r += input[pixel_idx].r * weight;
                    g += input[pixel_idx].g * weight;
                    b += input[pixel_idx].b * weight;
                }
            }
            
            int out_idx = y * width + x;
            output[out_idx].r = static_cast<uint8_t>(std::min(std::max(r, 0.0), 255.0));
            output[out_idx].g = static_cast<uint8_t>(std::min(std::max(g, 0.0), 255.0));
            output[out_idx].b = static_cast<uint8_t>(std::min(std::max(b, 0.0), 255.0));
        }
    }
    return output;
}

// Guardar imagen PPM
void savePPM(const std::string& filename, const std::vector<Pixel>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error al abrir el archivo " << filename << " para escritura.\n";
        return;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(Pixel));
    file.close();
}

int main() {
    // Configurar OpenMP para que use el máximo de hilos disponibles si se desea forzar
    // omp_set_num_threads(omp_get_max_threads());
    
    std::cout << "Hilos de OpenMP disponibles: " << omp_get_max_threads() << "\n\n";

    auto start_total = std::chrono::high_resolution_clock::now();

    // 1. TAREA A
    std::cout << "Iniciando Tarea A: Generacion de Mandelbrot 8K (" << WIDTH << "x" << HEIGHT << ")...\n";
    auto start_mandelbrot = std::chrono::high_resolution_clock::now();
    
    std::vector<Pixel> mandelbrot_img = generateMandelbrot(WIDTH, HEIGHT);
    
    auto end_mandelbrot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_mandelbrot = end_mandelbrot - start_mandelbrot;
    std::cout << "Mandelbrot generado en: " << diff_mandelbrot.count() << " segundos.\n";

    savePPM("mandelbrot_8k_original_omp.ppm", mandelbrot_img, WIDTH, HEIGHT);
    std::cout << "Imagen original guardada como 'mandelbrot_8k_original_omp.ppm'\n\n";

    // 2. TAREA B
    int blur_radius = 7; 
    std::cout << "Iniciando Tarea B: Convolucion de Desenfoque Gaussiano (Kernel " 
              << (blur_radius * 2 + 1) << "x" << (blur_radius * 2 + 1) << ")...\n";
    auto start_blur = std::chrono::high_resolution_clock::now();
    
    std::vector<Pixel> blurred_img = applyGaussianBlur(mandelbrot_img, WIDTH, HEIGHT, blur_radius);
    
    auto end_blur = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_blur = end_blur - start_blur;
    std::cout << "Filtro aplicado en: " << diff_blur.count() << " segundos.\n";

    savePPM("mandelbrot_8k_blurred_omp.ppm", blurred_img, WIDTH, HEIGHT);
    std::cout << "Imagen filtrada guardada como 'mandelbrot_8k_blurred_omp.ppm'\n\n";

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_total = end_total - start_total;
    std::cout << "Tiempo total de ejecucion: " << diff_total.count() << " segundos.\n";

    return 0;
}