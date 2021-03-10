import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','muted','no-latex'])
plt.rcParams.update({'font.size': 16})

E = -1*np.load("HF_E.npy")*27.2114*10**-3  #keV
x1 = np.linspace(-15.0, 15.0, 30)
x2 = np.linspace(-15.0, 15.0, 30)


fig = plt.figure(1, figsize=(10, 6))
plt.contourf(x1, x2, E, levels=15, cmap='gnuplot')
plt.colorbar(label="E (keV)")
plt.ylabel(r"$\chi_1$")
plt.xlabel(r"$\chi_2$")
plt.savefig("HF_Energy.png")
plt.savefig("HF_Energy.eps")