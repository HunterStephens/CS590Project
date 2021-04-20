import numpy as np
import os
import psi4
import time
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

n_res = [5, 10, 20, 30, 50, 100]
# --- create directory ---
pth = f"./data/output/n_res/"
if not os.path.exists(pth):
    os.makedirs(pth)

tHF = np.zeros(len(n_res))
tMP2 = np.zeros(len(n_res))
tDFT = np.zeros(len(n_res))

for i,n in enumerate(n_res):
    # --- load file ---
    seq_str = ""
    for j in range(n):
        seq_str = seq_str + "T"
    f = open(f"./data/psi4files/peptides/{seq_str}.txt","r")
    mol_string = f.read()
    start = time.time()
    HF_energy(mol_string,'cc-pvdz')
    end = time.time()
    tHF[i] = end-start
    print(tHF[i])



