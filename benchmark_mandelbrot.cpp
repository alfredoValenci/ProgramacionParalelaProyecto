#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>

struct Pixel { uint8_t r, g, b; };

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 1000;

// Función Mandelbrot adaptada para leer el scheduler en tiempo de ejecución
void benchmarkMandelbrot(const std::string& name) {
    std::vector<Pixel> image(WIDTH * HEIGHT);
    double min_re = -2.5, max_re = 1.0;
    double min_im = -1.0, max_im = 1.0;
    double re_factor = (max_re - min_re) / (WIDTH - 1);
    double im_factor = (max_im - min_im) / (HEIGHT - 1);

    auto start = std::chrono::high_resolution_clock::now();

    // El schedule se define antes de llamar a la función usando omp_set_schedule
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; ++y) {
        double c_im = max_im - y * im_factor;
        for (int x = 0; x < WIDTH; ++x) {
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
            
            int idx = y * WIDTH + x;
            if (isInside) {
                image[idx] = {0, 0, 0};
            } else {
                image[idx] = {static_cast<uint8_t>((n * 9) % 256), 
                              static_cast<uint8_t>((n * 5) % 256), 
                              static_cast<uint8_t>((n * 13) % 256)};
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << std::left << std::setw(25) << name 
              << ": " << diff.count() << " segundos.\n";
}

int main() {
    std::cout << "--- Benchmark de Schedulers OpenMP en Mandelbrot 8K ---\n";
    std::cout << "Hilos disponibles: " << omp_get_max_threads() << "\n\n";

    std::vector<int> chunk_sizes = {1, 16, 64, 256, 1024};

    // 1. Prueba STATIC
    omp_set_schedule(omp_sched_static, 0); // 0 deja que OpenMP decida los bloques
    benchmarkMandelbrot("Static (Default)");

    std::cout << "\n";

    // 2. Pruebas DYNAMIC
    for (int chunk : chunk_sizes) {
        omp_set_schedule(omp_sched_dynamic, chunk);
        benchmarkMandelbrot("Dynamic (Chunk " + std::to_string(chunk) + ")");
    }

    std::cout << "\n";

    // 3. Pruebas GUIDED
    for (int chunk : chunk_sizes) {
        omp_set_schedule(omp_sched_guided, chunk);
        benchmarkMandelbrot("Guided (Chunk " + std::to_string(chunk) + ")");
    }

    return 0;
}