import numpy as np
import psi4

# --- calculate energy surfaces ---
chi1 = np.load("./chi1.npy")
chi2 = np.load("./chi2.npy")
nr = len(chi1)
nc = len(chi2)
EHF = np.zeros((nr, nc))
EDFT = np.zeros((nr, nc))
ECC = np.zeros((nr, nc))
psi4.core.be_quiet()
counter = 0
for i in range(len(chi1)):
    for j in range(len(chi2)):
        # --- load file ---
        f = open(f"./data/psi4files/cys_{chi1[i]*180/np.pi}_{chi2[j]*180/np.pi}.txt", "r")
        mol_string = f.read()
        mol = psi4.geometry(mol_string)
        psi4.set_options({'reference': 'uhf'})
        EHF[i, j] = psi4.energy('scf/cc-pvdz')
        print(f"\r>> Completed {round(counter/31/31*100,2)}%", end='')
        #print(f"EHF[{i},{j}]={EHF[i,j]}")
        np.save("EHF.npy", EHF)
        counter = counter + 1

