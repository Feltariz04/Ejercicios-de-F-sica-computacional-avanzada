import numpy as np
import matplotlib.pyplot as plt


#Parametros de la señal
N = 10000
t_obs = 5 # tiempo de observación en segundos
fm = N / t_obs # frecuencia de muestreo
tn = np.arange(N) / fm # vector de tiempo

f = 250 # frecuencia de la señal en Hz

# Construcción de la señal
y = np.cos(2 * np.pi * f * tn)

# Calculando la DFT Normalizada
F = np.fft.fft(y) / N
freqs = np.fft.fftfreq(N, d=1/fm) # frequency

#Grafico de la parte real de la DFT
plt.figure(figsize=(10, 4))
plt.plot(freqs, F.real)
plt.xlim(-400, 400)
plt.title("Parte real de la DFT Normalizada")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Parte real")    
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()