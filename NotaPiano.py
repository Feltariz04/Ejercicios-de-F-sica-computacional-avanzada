# Nota musical de un piano

import numpy as np
import matplotlib.pyplot as plt

#Para llamar los datos
Piano = np.genfromtxt("piano.txt")

fs = 44100 # Hertz
N = len(Piano)
dt = 1/ fs
T = N * dt


#Obtengo los valores de los datos
y = Piano
t = np.arange(N) * dt

Y = np.fft.rfft(y)
freq = np.fft.rfftfreq(N, d=dt)

Potencia = np.abs(Y)**2

potenciaMax = 0
frecuenciaMax = 0

for i in range(1, len(Potencia)):
    if Potencia[i] > potenciaMax:
        potenciaMax = Potencia[i]
        frecuenciaMax = freq[i]

print(f"La frecuencia obtenida es {frecuenciaMax} (Hz), lo cual se puede acercar a la nota musical Do5")


# Grafica 

fig, axes = plt.subplots(3, 1, figsize = (10, 8))

fig.suptitle("Análisis de una nota musical de piano", fontsize=15)

#Grafico de la señal
axes[0].plot(t, y, color="royalblue", linewidth=0.7)
axes[0].set_title("Señal en el dominio del tiempo")
axes[0].set_ylabel("Amplitud")
axes[0].set_xlabel("Tiempo (s)")
axes[0].grid(True, alpha=0.3)

#Grafico de DFT
amplitud = np.abs(Y)

axes[1].plot(freq, amplitud, color="darkorange", linewidth=0.8)
axes[1].set_title("Transformada discreta de Fourier (DFT)")
axes[1].set_ylabel("Amplitud espectral")
axes[1].set_xlabel("Frecuencia (Hz)")
axes[1].set_xlim(0, 2000)
axes[1].grid(True, alpha=0.3)

#Grafico de la potencia
axes[2].plot(freq, Potencia, color="seagreen", linewidth=0.8)
axes[2].axvline(
    frecuenciaMax,
    color="red",
    linestyle="--",
    label=f"Frecuencia dominante: {frecuenciaMax:.2f} Hz"
)
axes[2].set_title("Espectro de potencia")
axes[2].set_ylabel(r"$|Y(f)|^2$")
axes[2].set_xlabel("Frecuencia (Hz)")
axes[2].set_xlim(0, 2000)
axes[2].legend()
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()