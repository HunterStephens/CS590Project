import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','muted','no-latex'])
plt.rcParams.update({'font.size': 16})
'''
chi1 = np.load("chi1.npy")
chi2 = np.load("chi2.npy")
EHF = np.load("EHF.npy")
EDFT = np.load("EDFT.npy")
EMP2 = np.load("EMP2.npy")

# -- plot energy surface ---
fig = plt.figure(1, figsize=(35, 6))
ax1 = plt.subplot(1, 3, 1)
ax1.set_title("Hartree-Fock")
plt.contourf(chi1, chi2, EHF, levels=20,cmap='RdYlBu_r')
plt.colorbar(label=r'$E_h$')
plt.xlabel(r"$\chi_2$")
plt.ylabel(r"$\chi_1$")

ax2 = plt.subplot(1, 3, 2)
ax2.set_title("DFT")
plt.contourf(chi1, chi2, EDFT, levels=20,cmap='RdYlBu_r')
plt.colorbar(label=r'$E_h$')
plt.xlabel(r"$\chi_2$")
plt.ylabel(r"$\chi_1$")



ax2 = plt.subplot(1, 3, 3)
ax2.set_title("MP2")
plt.contourf(chi1, chi2, EMP2, levels=20, cmap='RdYlBu_r')
plt.colorbar(label=r'$E_h$')
plt.xlabel(r"$\chi_2$")
plt.ylabel(r"$\chi_1$")
plt.savefig("E_surface.png")
plt.show()
'''

chi = np.load("chi.npy")
EHF1 = np.load("EHF1.npy")
EMP21 = np.load("EMP21.npy")


# -- plot energy surface ---
colors = ["#1b9e77", "#d95f02", "#7570b3"]
axd = plt.figure(figsize=(12, 8)).subplot_mosaic(
    """
    A
    A
    B
    """
)
axd['A'].plot(chi, EHF1 - np.amin(EHF1), linewidth=2, color=colors[0], label="Hartree-Fock")
axd['A'].plot(chi, EMP21 - np.amin(EMP21), linewidth=2, color=colors[1], label="MP2")
plt.legend()
axd['A'].set_ylabel(r"$E-E_0 \ (h)$")

axd['B'].plot(chi, EHF1, linewidth=2, color=colors[0], label="Hartree-Fock")
axd['B'].plot(chi, EMP21, linewidth=2, color=colors[1], label="MP2")

axd['B'].set_xlabel(r"$\chi$")
axd['B'].set_ylabel(r"$E \ (h)$")
axd['B'].set_ylim(-721, -718.5)
plt.savefig("E_conf_prof.png")
plt.show()


