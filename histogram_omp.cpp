#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Asumiendo que struct Pixel y la imagen ya existen
// struct Pixel { uint8_t r, g, b; };

// MÉTODO 1: Exclusión Mutua (Atomic)
std::vector<int> calculateHistogramAtomic(const std::vector<Pixel>& image) {
    std::vector<int> histogram(256, 0);
    
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < image.size(); ++i) {
        // Cálculo rápido de luminosidad (promedio de canales)
        int luminance = (image[i].r + image[i].g + image[i].b) / 3;
        
        // Exclusión mutua: asegura que solo un hilo modifique el bin a la vez
        #pragma omp atomic
        histogram[luminance]++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Histograma (Atomic) completado en: " << diff.count() << " segundos.\n";
    
    return histogram;
}

// MÉTODO 2: Reducción (Variables estrictamente locales implícitas)
std::vector<int> calculateHistogramReduction(const std::vector<Pixel>& image) {
    std::vector<int> histogram(256, 0);
    
    auto start = std::chrono::high_resolution_clock::now();

    // OpenMP 4.5+ permite reducción directa sobre arreglos en C/C++
    // Esto crea copias locales del arreglo por cada hilo y las suma al final
    #pragma omp parallel for reduction(+:histogram[:256])
    for (size_t i = 0; i < image.size(); ++i) {
        int luminance = (image[i].r + image[i].g + image[i].b) / 3;
        histogram[luminance]++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Histograma (Reduction) completado en: " << diff.count() << " segundos.\n";
    
    return histogram;
}
