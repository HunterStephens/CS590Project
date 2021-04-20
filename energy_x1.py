import numpy as np
import os
import psi4
psi4.core.be_quiet()

def HF_energy(mol_string, basis):
    psi4.geometry(mol_string)
    psi4.set_options({'reference': 'uhf'})
    return psi4.energy(f'scf/{basis}')

def DFT_energy(mol_string, basis):
    psi4.geometry(mol_string)
    psi4.set_options({'reference': 'uks'})
    return psi4.energy(f'b3lyp/{basis}')

def MP2_energy(mol_string, basis):
    psi4.geometry(mol_string)
    psi4.set_options({'reference': 'uhf'})
    psi4.set_options({'freeze_core': True})
    psi4.energy(f'mp2/{basis}')





# --- Energy profile calculations ---
chi = np.load("./chi.npy")
acids = ['cys']
for acid in acids:
    # --- create directory ---
    pth = f"./data/output/{acid}"
    if not os.path.exists(pth):
        os.makedirs(pth)

    EHF = np.zeros(len(chi))
    EMP2 = np.zeros(len(chi))
    EDFT = np.zeros(len(chi))
    for i in range(len(chi)):
        # --- load file ---
        f = open(f"./data/psi4files/{acid}/{acid}_{chi[i]}.txt", "r")
        mol_string = f.read()
        #EHF[i] = HF_energy(mol_string, 'cc-pvdz')
        #EMP2[i] = MP2_energy(mol_string, 'cc-pvdz')
        EDFT[i] = DFT_energy(mol_string, 'cc-pvdz')
        np.save(f"{pth}/EDFT.npy", EDFT)
        #np.save(f"{pth}/EHF.npy", EHF)
        #np.save(f"{pth}/EMP2.npy", EMP2)
        print(f"\r>> Completed {round(i / len(chi) * 100, 2)}%: {acid}", end='')
    print()

