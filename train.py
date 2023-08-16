import jax.numpy as np
from jax import grad

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

print(basis_list[1][2])
kmesh = np.load('data/kmesh.npy')
eigs  = np.load('data/eigs.npy')
length = kmesh.shape[0]
print(eigs[1], kmesh[1])
print(kmesh.shape, eigs.shape)

weights = np.zeros(7)
print(weights)


def make_hamk(weights, kpnt):
    
    kx = kpnt[0]
    ky = kpnt[1]
    kz = kpnt[2]

    hamk = (weights[0]+weights[1]*(kx**2+ky**2)+weights[2]*kz**2)*basis_list[0, 0]+(weights[3]+weights[4]*(kx**2+ky**2)+weights[5]*kz**2)*basis_list[3, 0]+weights[6]*(kx*basis_list[1, 3]-ky*basis_list[2, 0])

    return hamk


def training_loss(weights):

    loss = 0

    for i in range(length):
        hamk = make_hamk(weights, kmesh[i])
        w, v = np.linalg.eigh(hamk)
        loss += np.linalg.norm(w-eigs[i])

    return loss

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

print(training_gradient_fun(weights))
print("Initial loss:", training_loss(weights))
for i in range(100):
    print("i:", i)
    weights -= training_gradient_fun(weights) * 0.0001
    print(training_loss(weights))

print(weights)
print("Trained loss:", training_loss(weights))
