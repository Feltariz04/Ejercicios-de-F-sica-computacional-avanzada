import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 3
w0 = np.pi * 2 / T
k = 1

#Funcion periodica original
def f_base(t):    
    tau = (np.asarray(t) + 1) % T - 1
    return 2 * tau + 4


# coeficiente de la serie de fuorier
w = w0 * k

a0 = (2/T)* quad(lambda x: f_base(x), -1, 2)[0]

a1 = (2/T)* quad(lambda x: f_base(x)*np.cos(w * x), -1, 2)[0]

b1 = (2/T)* quad(lambda x: f_base(x)*np.sin(w * x), -1, 2)[0]


#Suma de de fourier
def sumar(x):
    fs = (a0/2) + a1 * np.cos(w * x) + b1 * np.sin(w * x)
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