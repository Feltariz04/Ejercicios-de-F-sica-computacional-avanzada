import numpy as np
import matplotlib.pyplot as plt

#Parametros de la señal
N = 1000
fs = N/5 # frecuencia de muestreo
tn = np.arange(N) / fs # 


f1 = 10 # frecuencia 1 Hz
f2 = 50 # frecuencia 2 Hz


# Construcción de la señal
y = np.sin(2 * np.pi * f1 * tn) + 2 * np.cos(2 * np.pi * f2 * tn)

# Calculando la DFT Normalizada
F = np.fft.fft(y) / N
freqs = np.fft.fftfreq(N, d=1/fs) # frequency bins

#Ordenar las frecuencias y la DFT para que estén en orden creciente
freqs = np.fft.fftshift(freqs)
F = np.fft.fftshift(F)

#Espectro de potencia
P = np.abs(F)**2

# Plot the the part real of the DFT
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.stem(freqs, F.real)
plt.xlim(-75, 75)
plt.title("Parte real de la DFT Normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Parte real")

# Plot the imaginary part of the DFT
plt.subplot(3, 1, 2)
plt.stem(freqs, F.imag)
plt.xlim(-75, 75)
plt.title("Parte imaginaria de la DFT Normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Parte imaginaria")

# Plot the power spectrum
plt.subplot(3, 1, 3)
plt.stem(freqs, P)
plt.xlim(-75, 75)
plt.title("Espectro de potencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")

plt.tight_layout()
plt.show()