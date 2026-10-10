
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 5
w0 = 2 * np.pi / T

# 1. Funcion original en un periodo [5, 10)
def f_base(x):
    if 5 <= x < 7:
        return x - 5
    elif 7 <= x < 9:
        return 2.0
    else:
        return 10 - x

# 2. Extension periodica
def f_periodica(x):
    r = 5 + np.mod(x - 5, T)
    return np.where(
        r < 7,
        r - 5,
        np.where(r < 9, 2.0, 10 - r)
    )

# 3. Calculo de los coeficientes
a0 = (1 / T) * (
    quad(lambda x: x - 5, 5, 7)[0]
    + quad(lambda x: 2.0, 7, 9)[0]
    + quad(lambda x: 10 - x, 9, 10)[0]
)

def calcular_ak(k):
    w = k * w0
    integral = (
        quad(lambda x: (x - 5) * np.cos(w*x), 5, 7)[0]
        + quad(lambda x: 2 * np.cos(w*x), 7, 9)[0]
        + quad(lambda x: (10 - x) * np.cos(w*x), 9, 10)[0]
    )
    return (2 / T) * integral

def calcular_bk(k):
    w = k * w0
    integral = (
        quad(lambda x: (x - 5) * np.sin(w*x), 5, 7)[0]
        + quad(lambda x: 2 * np.sin(w*x), 7, 9)[0]
        + quad(lambda x: (10 - x) * np.sin(w*x), 9, 10)[0]
    )
    return (2 / T) * integral

Nmax = 50
ks = np.arange(1, Nmax + 1)

ak = np.array([calcular_ak(k) for k in ks])
bk = np.array([calcular_bk(k) for k in ks])

print(f"a0 = {a0:.6f}")

for k in range(1, 6):
    print(f"k={k}: ak={ak[k-1]:.6f}, "
          f"bk={bk[k-1]:.6f}")

# 4. Suma parcial de Fourier
def S_fourier(x, N):
    S = np.full_like(np.asarray(x, dtype=float), a0)

    for k in range(1, N + 1):
        S += (
            ak[k-1] * np.cos(k*w0*x)
            + bk[k-1] * np.sin(k*w0*x)
        )

    return S

# 5. Grafica de la funcion y sumas parciales
x = np.linspace(0, 15, 2000)

plt.figure(figsize=(11, 6))
plt.plot(x, f_periodica(x), 'k', linewidth=2,
         label='Funcion periodica original')

for N in [5, 20, 50]:
    plt.plot(x, S_fourier(x, N), label=f'N = {N}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Serie de Fourier: sumas parciales')
plt.xlim(0, 15)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Decaimiento de los coeficientes
plt.figure(figsize=(9, 5))
plt.semilogy(ks, np.abs(ak), 'o-', markersize=3,
             label=r'$|a_k|$')
plt.semilogy(ks, np.abs(bk), 's-', markersize=3,
             label=r'$|b_k|$')

plt.xlabel('k')
plt.ylabel('Valor absoluto del coeficiente')
plt.title('Decaimiento de los coeficientes de Fourier')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
