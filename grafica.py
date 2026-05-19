import matplotlib.pyplot as plt

# ==========================================
# 1. TUS DATOS EMPÍRICOS (1 a 8 hilos)
# ==========================================
hilos = [1, 2, 3, 4, 5, 6, 7, 8]

# Tiempos totales extraídos de tus capturas
tiempos_totales = [29.1624, 15.0758, 10.3405, 8.60808, 9.64733, 8.76791, 8.21685, 7.88798]

# Tiempos desglosados
tiempos_mandel = [18.0623, 9.06431, 6.13087, 5.15382, 4.66806, 4.57943, 4.58596, 4.60723]
tiempos_conv = [10.6912, 5.5796, 3.80247, 3.047, 4.56656, 3.77301, 3.24319, 2.89887]

# ==========================================
# 2. CÁLCULO DE SPEEDUP
# ==========================================
# Fórmula: T_secuencial / T_paralelo
tiempo_secuencial = tiempos_totales[0]
speedup = [tiempo_secuencial / t for t in tiempos_totales]

# ==========================================
# 3. GENERACIÓN DE GRÁFICAS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- GRÁFICA 1: Tiempo de Ejecución vs Hilos ---
ax1.plot(hilos, tiempos_totales, marker='o', color='tab:red', linewidth=2, label='Tiempo Total')
ax1.plot(hilos, tiempos_mandel, marker='s', linestyle='--', color='tab:orange', label='Mandelbrot')
ax1.plot(hilos, tiempos_conv, marker='^', linestyle='--', color='tab:purple', label='Convolución')

ax1.set_title('Tiempo de Ejecución vs. Número de Hilos', fontsize=14)
ax1.set_xlabel('Número de Hilos', fontsize=12)
ax1.set_ylabel('Tiempo (segundos)', fontsize=12)
ax1.set_xticks(hilos)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- GRÁFICA 2: Speedup vs Hilos ---
ax2.plot(hilos, speedup, marker='o', color='tab:blue', linewidth=2, label='Speedup Real')
ax2.plot(hilos, hilos, linestyle='--', color='gray', label='Speedup Ideal (Lineal)')

ax2.set_title('Aceleración (Speedup) vs. Número de Hilos', fontsize=14)
ax2.set_xlabel('Número de Hilos', fontsize=12)
ax2.set_ylabel('Aceleración', fontsize=12)
ax2.set_xticks(hilos)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Ajustar y mostrar
plt.tight_layout()
plt.show()