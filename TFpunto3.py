import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 2
w0 = np.pi * 2 / T
k = 1

#Funcion periodica original
def f_base(t):    
    tau = (np.asarray(t) + T/2) % T - T/2
    return np.where(tau < 0, -0.5, 0.5)


# coeficiente de la serie de fuorier
w = w0 * k
w2 = w0 * (k +1)

a0 = (2/T)* (quad(lambda x: f_base(x), -1, 0)[0] 
             + quad(lambda x: f_base(x), 0, 1)[0])

b1 = (2/T)* (quad(lambda x: f_base(x)*np.sin(w * x), -1, 0)[0] 
             + quad(lambda x: f_base(x)*np.sin(w * x), 0, 1)[0])

b2 = (2/T)* (quad(lambda x: f_base(x)*np.sin(w2 * x), -1, 0)[0] 
             + quad(lambda x: f_base(x)*np.sin(w2 * x), 0, 1)[0])

#Suma de de fourier
def sumar(x):
    fs = (a0/2) + b1 * np.sin(w * x) + b2 * np.sin(w2 * x)
    return fs

#Graficar la funcion

x = np.linspace(-4, 4, 1000)

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