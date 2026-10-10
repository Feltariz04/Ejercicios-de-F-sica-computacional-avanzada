import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 5
w0 = np.pi * 2 / T
k = 1

#Funcion periodica original
def f_base(t):    
    tau = (np.asarray(t) + 1) % T - 1
    return tau**2


# coeficiente de la serie de fuorier
w = w0 * k
w2= w0 * (k+1)

a0 = (2/T)* quad(lambda x: f_base(x), 0, 5)[0]

a1 = (2/T)* quad(lambda x: f_base(x)*np.cos(w * x), 0, 5)[0]

a2 = (2/T)* quad(lambda x: f_base(x)*np.cos(w2 * x), 0, 5)[0]


#Suma de de fourier
def sumar(x):
    fs = (a0/2) + a1 * np.cos(w * x) + a2 * np.cos(w2 * x)
    return fs

#Graficar la funcion

x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(8,5))

plt.plot(x, f_base(x), 'k--', label='Funcion condicionada')

plt.plot(x, sumar(x), 'r-', label = 'Funcion Fourier')

plt.title('Serie de Fourier de f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()