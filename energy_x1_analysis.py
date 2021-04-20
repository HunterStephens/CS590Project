import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use(['science','muted','no-latex'])
plt.rcParams.update({'font.size': 16})

acid = 'cys'

chi = np.load(f"./data/output/{acid}/chi.npy")
EHF1 = np.load(f"./data/output/{acid}/EHF.npy")
EMP21 = np.load(f"./data/output/{acid}/EMP2.npy")
EDFT1 = np.load(f"./data/output/{acid}/EDFT.npy")


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
axd['A'].plot(chi, EDFT1 - np.amin(EDFT1), linewidth=2, color=colors[2], label="DFT")
plt.legend()
axd['A'].set_ylabel(r"$E-E_0 \ (h)$")

axd['B'].plot(chi, EHF1, linewidth=2, color=colors[0], label="Hartree-Fock")
axd['B'].plot(chi, EMP21, linewidth=2, color=colors[1], label="MP2")
axd['B'].plot(chi, EDFT1, linewidth=2, color=colors[2], label="DFT")

axd['B'].set_xlabel(r"$\chi_1$")
axd['B'].set_ylabel(r"$E \ (h)$")
axd['B'].set_ylim(-723, -718.5)
# --- create directory ---
pth = f"./plots/{acid}"
if not os.path.exists(pth):
    os.makedirs(pth)
plt.savefig(f"{pth}/E.png")
plt.show()


