import numpy as np
import psi4
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def Rx(theta):
    theta = theta * np.pi / 180
    R = np.zeros((3, 3))
    R[0, 0] = 1.0
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(theta)
    R[2, 1] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R


def Ry(theta):
    theta = theta * np.pi / 180
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(theta)
    R[0, 2] = np.sin(theta)
    R[1, 1] = 1
    R[2, 0] = -np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R


def Rz(theta):
    theta = theta * np.pi / 180
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    R[2, 2] = 1
    return R


class base:
    def __init__(self):
        self.coords = np.zeros((3, 1))
        self.atoms = []


class side_chain:
    def __init__(self):
        self.coords = np.zeros((3, 1))
        self.atoms = []


class amino_acid:
    def __init__(self):
        self.base = base()
        self.side_chain = side_chain()

    def getBase(self, fname):
        num_atoms = sum(1 for line in open(fname))
        self.base.coords = np.zeros((3, num_atoms))
        with open(fname) as f:
            for i, l in enumerate(f):
                splts = l.rstrip().split(" ")
                self.base.atoms.append(splts[0])
                coords = np.array([float(splts[1]), float(splts[2]), float(splts[3])])
                self.base.coords[:, i] = coords

    def getSideChain(self, fname):
        num_atoms = sum(1 for line in open(fname))
        self.side_chain.coords = np.zeros((3, num_atoms))
        with open(fname) as f:
            for i, l in enumerate(f):
                splts = l.rstrip().split(" ")
                self.side_chain.atoms.append(splts[0])
                coords = np.array([float(splts[1]), float(splts[2]), float(splts[3])])
                self.side_chain.coords[:, i] = coords

    def center(self):
        # --- find alpha carbon ---
        found = False
        xnot = 0
        ynot = 0
        znot = 0
        for i in range(len(self.base.atoms)):
            if self.base.atoms[i] == 'Ca':
                xnot = self.base.coords[0, i]
                ynot = self.base.coords[1, i]
                znot = self.base.coords[2, i]
                found = True
                break

        if not found:
            print("No alpha carbon found\nNo centering performed")

        self.base.coords[0, :] = self.base.coords[0, :] - xnot
        self.base.coords[1, :] = self.base.coords[1, :] - ynot
        self.base.coords[2, :] = self.base.coords[2, :] - znot
        self.side_chain.coords[0, :] = self.side_chain.coords[0, :] - xnot
        self.side_chain.coords[1, :] = self.side_chain.coords[1, :] - ynot
        self.side_chain.coords[2, :] = self.side_chain.coords[2, :] - znot

    def show(self, save=False, name="acid.png", show=True):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.base.coords[0, :], self.base.coords[1, :], self.base.coords[2, :],
                     c="green", s=75, label="Base")
        ax.scatter3D(self.side_chain.coords[0, :], self.side_chain.coords[1, :],
                     self.side_chain.coords[2, :], c="red", s=75, label="Side-chain")
        ax.view_init(elev=15, azim=0)
        ax.legend()
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        ax.set_zlim(-2,2)
        if save:
            plt.savefig(name)
        if show:
            plt.show()
        plt.close(fig)

    def flexChain(self, theta, phi, psi):
        # --- create flex acid ---
        flex_acid = amino_acid()
        T = np.matmul(Ry(phi), Rx(theta))
        T = np.matmul(Rz(psi), T)
        flex_acid.side_chain.coords = np.matmul(T, self.side_chain.coords)
        flex_acid.side_chain.atoms = self.side_chain.atoms
        flex_acid.base = self.base
        return flex_acid

    def HF_energy(self):
        psi4.set_memory('3 GB')
        psi4.core.be_quiet()
        geom_string = """\n"""
        for i in range(len(self.base.atoms)):
            if self.base.atoms[i] == 'Ca':
                atom_label = 'C'
            else:
                atom_label = self.base.atoms[i]

            geom_string = geom_string + f"{atom_label} {self.base.coords[0,i]}  \
                          {self.base.coords[1,i]} {self.base.coords[2,i]}\n"

        for i in range(len(self.side_chain.atoms)):
            if self.side_chain.atoms[i] == 'Ca':
                atom_label = 'C'
            else:
                atom_label = self.side_chain.atoms[i]

            geom_string = geom_string + f"{atom_label} {self.side_chain.coords[0,i]}  \
                          {self.side_chain.coords[1,i]} {self.side_chain.coords[2,i]}\n"

        mol = psi4.geometry(geom_string)
        psi4.set_options({'reference': 'uhf'})
        return psi4.energy('scf/cc-pvdz')



# --- create and read in amino acid ---
acid = amino_acid()
acid.getBase("cys_base.xyz")
acid.getSideChain("cys_res.xyz")
acid.center()


# --- perform baseline energy calcs ---

x1 = np.linspace(-15.0, 15.0, 30)
x2 = np.linspace(-15.0, 15.0, 30)
E = np.zeros((30, 30))


for i in range(30):
    for j in range(30):
        flex = acid.flexChain(x1[i], x2[j], 0.0)
        E[i,j] = flex.HF_energy()

np.save("HF_E.npy", E)






