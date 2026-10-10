import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 4
w0 = 2 * np.pi / T

# Funcion original con un periodo 4
def f_base(x):
    fx = -0.5*x - 1
    return fx

# Extension periodica
def f_periodica(x):
    r = -4 + np.mod(x + 4, T)
    return -0.5*r -1

# Calculo de solo el coeficiente de bk 
# a0 y ak no son necesarios para este ejercicio ya que su valor es 0

def calcular_bk(k):
    w = k * w0
    integral = quad(lambda x: f_base(x) * np.sin(w*x), -4, 0)[0]
    return (2 / T) * integral

#Definimos el rango de k para calcular los coeficientes bk
ks = np.arange(1, 9)
bk = np.array([calcular_bk(k) for k in ks])

#Suma de la serie de Fourier
def serie_fourier(x, N):
    s = 0
    for k in range(1, N + 1):
        s += bk[k - 1] * np.sin(k * w0 * x)
    return s


# Graficamos la función original y la serie de Fourier para diferentes valores de N

x = np.linspace(-4, 7.9, 1000)

plt.figure(figsize=(8, 5))
plt.plot(x, f_periodica(x), 'k--', label='función original')

plt.plot(x, serie_fourier(x, 8), 'r', label=f'suma de Fourier N=8')

plt.title('Serie de Fourier de f(x) = -0.5x - 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-1,1)
plt.grid(False)
plt.legend()
plt.tight_layout()

plt.show()