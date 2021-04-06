import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','muted','no-latex'])
plt.rcParams.update({'font.size': 16})

Ehf = np.load("./data/cysteine/HF_E.npy")
Edft = np.load("./data/cysteine/DFT_E.npy")
x1 = np.linspace(-15.0, 15.0, 30)
x2 = np.linspace(-15.0, 15.0, 30)


fig = plt.figure(1, figsize=(20, 6))
ax1 = plt.subplot(1, 2, 1)
ax1.set_title("Hartree-Fock")
plt.contourf(x1, x2, Ehf, levels=15, cmap='gnuplot')
plt.colorbar(label="$E_h$")
plt.ylabel(r"$\chi_1$")
plt.xlabel(r"$\chi_2$")

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("B3LYP - DFT")
plt.contourf(x1, x2, Edft, levels=15, cmap='gnuplot')
plt.colorbar(label="$E_h$")
plt.ylabel(r"$\chi_1$")
plt.xlabel(r"$\chi_2$")


plt.savefig("./plots/E_surface.png")
plt.savefig("./plots/E_sruface.eps")


# --- get minimum energy ---
hf_min_inds = np.argmin(Ehf)
hf_min_inds = np.unravel_index(hf_min_inds, Ehf.shape)

dft_min_inds = np.argmin(Edft)
dft_min_inds = np.unravel_index(dft_min_inds, Ehf.shape)



fig = plt.figure(2, figsize=(12, 8))
ax = plt.subplot(1, 1, 1)
ax.plot(x1, Ehf[hf_min_inds[0], :], color="#1b9e77", \
        label=r"$HF, \chi_1 =$"+f"{np.round(x1[hf_min_inds[1]],decimals=2)}", \
        linewidth=1.5)
ax.set_ylabel("$E_h$, Hartree-Fock")
ax.set_xlabel("$\chi_2$")
ax2 = ax.twinx()
ax2.plot(x1, Edft[dft_min_inds[0], :], color="#7570b3", \
         label=r"$DFT, \chi_1 =$"+f"{np.round(x1[dft_min_inds[1]],decimals=2)}", \
         linewidth=1.5)
ax2.set_ylabel("$E_h$, B3LYP - DFT")
fig.legend(loc="upper center")
plt.savefig("./plots/E_profile.png")
plt.savefig("./plots/E_profile.eps")
plt.show()

fig = plt.figure(3, figsize=(12, 8))
ax = plt.subplot(1, 1, 1)
diff = Edft - Ehf
error = np.abs(np.divide(diff, Edft))
plt.pcolormesh(x1, x2, error, cmap='gnuplot',alpha=0.6)
plt.colorbar(label="$\sigma_{E_h}$")
plt.contour(x1, x2, Ehf, levels=15, c="black")
plt.ylabel(r"$\chi_1$")
plt.xlabel(r"$\chi_2$")
plt.savefig("./plots/HF_on_error.png")
plt.savefig("./plots/HF_on_error.eps")
plt.show()


# --- get mean error ---
mean_error = np.mean(error)
print(mean_error)

# --- get distance from min ---
hf_min_x1 = x1[hf_min_inds[0]]
hf_min_x2 = x1[hf_min_inds[1]]

dft_min_x1 = x1[dft_min_inds[0]]
dft_min_x2 = x1[dft_min_inds[1]]

dist = (hf_min_x1-dft_min_x1)**2 + (hf_min_x2-dft_min_x2)**2
dist = dist**0.5
print(dist)