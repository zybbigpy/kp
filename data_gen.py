import numpy as np
#import matplotlib.pyplot as plt


sigma0 = np.array([[1, 0],[0, 1]])
sigmax = np.array([[0, 1],[1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

sigma_list =[sigma0, sigmax, sigmay, sigmaz]

s0 = np.array([[1, 0],[0, 1]])
sx = np.array([[0, 1],[1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
s_list = [s0, sx, sy, sz]


basis_list  = [np.kron(sigma, s) for sigma in sigma_list for s in s_list]
basis_list = np.array(basis_list)
basis_list = basis_list.reshape(4, 4, 4, 4)

print(basis_list[1, 3], basis_list[2, 0])

def set_tb_disp_kmesh(n_k: int, high_symm_pnts: dict) -> tuple:

    num_sec = len(high_symm_pnts)
    num_kpt = n_k*(num_sec-1)
    length = 0

    klen = np.zeros((num_sec), float)
    kline = np.zeros((num_kpt+1), float)
    kmesh = np.zeros((num_kpt+1, 3), float)
    ksec = []
    
    for key in high_symm_pnts:
        ksec.append(high_symm_pnts[key])

    for i in range(num_sec-1):
        vec = ksec[i+1]-ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i+1] = klen[i]+length

        for ikpt in range(n_k):
            kline[ikpt+i*n_k] = klen[i]+ikpt*length/n_k
            kmesh[ikpt+i*n_k] = ksec[i]+ikpt*vec/n_k
    kline[num_kpt] = kline[(num_sec-2)*n_k]+length
    kmesh[num_kpt] = ksec[-1]

    return (kline, kmesh)


C0 = -0.0145
C1 = 10.59
C2 = 11.5
M0 = -0.0205
M1 = 18.77
M2 = 13.5
A  = -0.889


def hamk_bulk(kpnt):

    # kpnt is 3d numpy.ndarray
    kx = kpnt[0]
    ky = kpnt[1]
    kz = kpnt[2]

    hamk = (C0+C2*(kx**2+ky**2)+C1*kz**2)*basis_list[0, 0]+(M0+M2*(kx**2+ky**2)+M1*kz**2)*basis_list[3, 0]+A*(kx*basis_list[1, 3]-ky*basis_list[2, 0])

    return hamk


# primitive lattice 
# angstrom
a = 12.9077420000000007
c = 25.9834260000000015
a1 = np.array([-a/2, a/2, c/2])
a2 = np.array([a/2, -a/2, c/2])
a3 = np.array([a/2, a/2, -c/2])

vol = np.dot(a1, np.cross(a2, a3))

b1 = 2*np.pi*(np.cross(a2, a3))/vol
b2 = 2*np.pi*(np.cross(a3, a1))/vol
b3 = 2*np.pi*(np.cross(a1, a2))/vol

Gamma = np.array([0, 0, 0])
X = 1/2*b3
Z = 1/2*b1+1/2*b2-1/2*b3

print("Z point", Z)

n_k = 300
high_symm_points = {'Z':Z, 'Gamma':Gamma, 'X':X}
kline, kmesh = set_tb_disp_kmesh(n_k, high_symm_points)
print("kline min, max", np.min(kline), np.max(kline))
all_hamk = [hamk_bulk(kpnt) for kpnt in kmesh]
eig_vals, _ = np.linalg.eigh(all_hamk)
band = eig_vals.shape[1]

np.save("./data/kmesh.npy", kmesh)
np.save("./data/eigs.npy", eig_vals)

# fig, ax = plt.subplots(figsize=(6, 6))
# for i in range(band):
#     if i==0:
#         ax.plot(kline, eig_vals[:, i], label='$kp$', **{'ls':'-', 'color':'#4A90E2', 'lw':2.5})
#     else:
#         ax.plot(kline, eig_vals[:, i], **{'ls':'-', 'color':'#4A90E2', 'lw':2.5})


# ax.set_xticks([kline[0], kline[n_k], kline[2*n_k]])
# ax.set_xticklabels(["Z", "$\Gamma$", "X"])
# ax.set_xlim(0, kline[-1])
# ax.set_ylim(-0.40, 0.40)
# ax.set_ylabel("Engergy (eV)")
# ax.axvline(x=kline[0], color="grey", lw=0.8, ls='--')
# ax.axvline(x=kline[n_k], color="grey", lw=0.8, ls='--')
# ax.axvline(x=kline[2*n_k], color="grey", lw=0.8, ls='--')

# plt.show()