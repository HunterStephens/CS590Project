import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','muted','no-latex'])
plt.rcParams.update({'font.size': 16})

chi1 = np.load("chi1.npy")
chi2 = np.load("chi2.npy")
EHF = np.load("EHF.npy")

min = np.argmin(EHF)
print(min)

fig = plt.figure(1, figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
plt.contourf(chi1,chi2,EHF,levels=20)
plt.colorbar()
plt.show()