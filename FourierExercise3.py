import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 2
w0 = 2 * np.pi / T

# Funcion original con un periodo 2
def f_base(x):
    fx = np.power(x, 2)
    return fx


# Calculo de solo el coeficientes
a0 = (1 / T) * quad(lambda x: f_base(x), 1, 3)[0]

def calcular_ak(k):
    w = k * w0
    integral = quad(lambda x: f_base(x) * np.cos(w*x), 1, 3)[0]
    return (2 / T) * integral

def calcular_bk(k):
    w = k * w0
    integral = quad(lambda x: f_base(x) * np.sin(w*x), 1, 3)[0]
    return (2 / T) * integral

#Definimos el rango de k para calcular los coeficientes ak y bk
ks = np.arange(1, 9)
ak = np.array([calcular_ak(k) for k in ks])
bk = np.array([calcular_bk(k) for k in ks])

#Suma de la serie de Fourier
def serie_fourier(x, N):
    s = np.zeros_like(np.asarray(x, dtype=float)) + a0 / 2
    for k in range(1, N + 1):
        s += ak[k - 1] * np.cos(k * w0 * x) + bk[k - 1] * np.sin(k * w0 * x)
    return s

# Graficamos la función original y la serie de Fourier para diferentes valores de N
x = np.linspace(1, 3, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, [f_base(i) for i in x], label='Función original', linewidth=2)
plt.plot(x, [serie_fourier(i, 8) for i in x], label='Serie de Fourier (N=8)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Aproximación de la función original mediante la serie de Fourier')
plt.legend()
plt.grid(True)
plt.show()
