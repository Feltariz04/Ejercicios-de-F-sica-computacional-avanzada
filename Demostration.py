
import numpy as np
import SolutionSteelBar as ftsc

# Comprobacion de la condicion de estabilidad FTCS

C2 = ftsc.C2
dx = ftsc.dx
dt_ftcs = 0.02  # Paso temporal en horas

r = C2 * dt_ftcs / dx**2
dt_max = dx**2 / (2 * C2)

print(f"Numero de Fourier r = {r:.4f}")
print(f"Limite de estabilidad r <= 0.5: {r <= 0.5}")
print(f"Paso temporal utilizado = {dt_ftcs:.5f} h")
print(f"Paso temporal maximo = {dt_max:.5f} h")

if r <= 0.5:
    print("FTCS cumple la condicion de estabilidad.")
else:
    print("FTCS no cumple la condicion de estabilidad.")
