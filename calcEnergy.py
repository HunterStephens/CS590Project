import numpy as np
import psi4


psi4.core.be_quiet()


# ---- Energy surface calculations ---
'''
chi1 = np.load("./chi1.npy")
chi2 = np.load("./chi2.npy")
nr = len(chi1)
nc = len(chi2)
EHF = np.zeros((nr, nc))
EDFT = np.zeros((nr, nc))
EMP2 = np.zeros((nr, nc))
counter = 0
for i in range(len(chi1)):
    for j in range(len(chi2)):
        # --- load file ---
        f = open(f"./data/psi4files/cys_{chi1[i]}_{chi2[j]}.txt", "r")
        mol_string = f.read()
        mol = psi4.geometry(mol_string)
        psi4.set_options({'reference': 'uhf'})
        EHF[i, j] = psi4.energy('scf/cc-pvdz')
        print(f"\r>> Completed {round(counter/31/31*100,2)}%", end='')
        np.save("EHF.npy", EHF)
        counter = counter + 1
'''

'''
counter = 0
for i in range(len(chi1)):
    for j in range(len(chi2)):
        # --- load file ---
        f = open(f"./data/psi4files/cys_{chi1[i]}_{chi2[j]}.txt", "r")
        mol_string = f.read()
        mol = psi4.geometry(mol_string)
        psi4.set_options({'reference': 'uks'})
        EDFT[i,j]= psi4.energy('b3lyp/cc-pvdz')
        print(f"\r>> Completed {round(counter/31/31*100,2)}%", end='')
        np.save("EDFT.npy", EDFT)
        counter = counter + 1
'''

'''
counter = 0
for i in range(len(chi1)):
    for j in range(len(chi2)):
        # --- load file ---
        f = open(f"./data/psi4files/cys_{chi1[i]}_{chi2[j]}.txt", "r")
        mol_string = f.read()
        mol = psi4.geometry(mol_string)
        psi4.set_options({'reference': 'uhf'})
        psi4.set_options({'freeze_core': True})
        EMP2[i, j] = psi4.energy('mp2/cc-pvdz')
        print(f"\r>> Completed {round(counter/31/31*100,2)}%", end='')
        np.save("EMP2.npy", EMP2)
        counter = counter + 1
'''
# --- Energy profile calculations ---
chi = np.load("./chi.npy")
EHF1 = np.zeros(len(chi))
EDFT1 = np.zeros(len(chi))
EMP21 = np.zeros(len(chi))
'''
for i in range(len(chi)):
    # --- load file ---
    f = open(f"./data/psi4files/cys_{chi[i]}.txt", "r")
    mol_string = f.read()
    mol = psi4.geometry(mol_string)
    psi4.set_options({'reference': 'uhf'})
    EHF1[i] = psi4.energy('scf/cc-pvdz')
    print(f"\r>> Completed {round(i / len(chi) * 100, 2)}%", end='')
    np.save("EHF1.npy", EHF1)
'''

for i in range(len(chi)):
    # --- load file ---
    f = open(f"./data/psi4files/cys_{chi[i]}.txt", "r")
    mol_string = f.read()
    mol = psi4.geometry(mol_string)
    psi4.set_options({'reference': 'uhf'})
    psi4.set_options({'freeze_core': True})
    EMP21[i] = psi4.energy('mp2/cc-pvdz')
    print(f"\r>> Completed {round(i / len(chi) * 100, 2)}%", end='')
    np.save("EMP21.npy", EMP21)