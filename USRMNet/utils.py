##################################  Define Some Useful Functions  ###############################

import numpy as np
import torch as t
from decimal import *
import scipy
# import sympy
import math
# from pynverse import inversefunc
# import cvxpy as cp


def indicator(K):
    # This function is used to generate indicators to obtain a specific variable
    '''
    @K:             number of users
    '''
    return t.eye(5*K)


def gamma(p, H, W, sigmma):
    # This function is used to compute SINR of all users with a batchsize for downlink scenerio
    '''
    @p:             downlink power vector of all users, its dimmension is (batchsize, K)
    @H:             channel coefficients of a batchsize, its dimmension is (batchsize, K, Nt)
    @W:             beamforming vector matrix of a batchsize, its dimmension is (batchsize, Nt, K), where Nt
                    is the antenna number of BS
    @sigmma:        Gaussian noise, suppose the values of all users are the same. Its dimmension is (1,1)
    '''
    H_bar = H / sigmma                 
    HW_tmp = t.abs(t.bmm(H_bar, W))                                         #   (batchsize, K, K)
    # print("solution1",t.pow(t.abs(H_bar.conj()[0,0,:]@W[0,:,0]),2))
    # print("solution2:",t.pow(t.abs((H_bar[0,0].conj()@W[0,:,0])),2))
    # print("solution3",HW_tmp[0,0,0])
    # print("HW_tmp:",HW_tmp[0,0,0]**2)
    HW_abs_2 = t.pow(HW_tmp, 2)                                                          #   (batchsize, K, K)
    HW_abs_tmp = t.zeros(len(H), len(H[0]))                                          #   (batchsize, K)

    for i in range(len(HW_tmp)):
        HW_abs_tmp[i] = t.diag(HW_abs_2[i])

    gamma_top = p * HW_abs_tmp                                                          #   (batchsize, K)
    gamma_botom = t.sum(p.unsqueeze(-2) * HW_abs_2, -1) - gamma_top + 1                 #   (batchsize, K)
    gamma_output = gamma_top / gamma_botom                                              #   (batchsize, K)
    return gamma_output


def leftrow_gamma(q, H, W, sigmma):
    # This function is used to compute SINR of all users with a batchsize for uplink scenerio
    '''
    @q:             uplink power vector of all users, its dimmension is (batchsize, K)
    @H:             channel coefficients of a batchsize, its dimmension is (batchsize, Nt, K)
    @W:             beamforming vector matrix of a batchsize, its dimmension is (batchsize, Nt, K), where Nt is 
                    the antenna number of BS
    @sigmma:        Gaussian noise, suppose the values of all users are the same. Its dimmension is (1,1)
    '''
    H_bar = H / sigmma
    HW_tmp = t.zeros(len(H_bar), len(H_bar[0][0]), len(H_bar[0][0]))                    # (batchsize, K, K)
    
    for i in range(len(HW_tmp)):
        HW_tmp[i] = t.abs(t.mm(H_bar[i].T, W[i]))
    
    HW_abs_pow = t.pow(HW_tmp, 2)
    HW_abs_tmp = t.zeros(len(H), len(H[0][0]))                                          # (batchsize, K)

    for i in range(len(HW_tmp)):
        HW_abs_tmp[i] = t.diag(HW_abs_pow[i])

    gamma_top = q * HW_abs_tmp                                                          # (batchsize, K)
    gamma_botom = t.sum(q.unsqueeze(-1) * HW_abs_pow, -1) - gamma_top + 1               # (batchsize, K)
    gamma_output = gamma_top / gamma_botom                                              # (batchsize, K)
    return gamma_output


def B_func(n, z):
    # subfunction of W_func
    # n determins B_func takes the sum of the first n-1 items， ‘z’ is the variable of B_func
    tmp1 = 0
    for k in range(n):
        tmp1 = tmp1 + ( math.factorial(n-1+k) / ( math.factorial(k) *  \
              math.factorial(n-1-k))) * ((z/2)**k)
    return tmp1


def W_func(t1, t2, a, n):
	# where t1 = 2\vartheta, t2 = -2\vartheta, a = -4*β^2*\vartheta^2, n denotes W_func takes 
    # the sum of the first n items
	tmp = 0
	for k in range(1, n+1):
		tmp = tmp + (1.0 / (k * math.factorial(k))) * ((a * k * np.exp(-t1)) \
              / (t2-t1))**k * B_func(k, -2.0/(k*(t2-t1)))
	return t1 - tmp


def k_star(beta, vartheta):
    # this function is used to obtain the solution of equation e^k * (k-2u)(k+2u) = -4beta^2u^2 
    t1 = 2 * vartheta
    t2 = -2 * vartheta
    a = -4 * beta * beta * vartheta * vartheta
    k_star_output = W_func(t1, t2, a, 10)
    return k_star_output


def nu2(beta, vartheta):
    # this function is used to obatin the solution of R(\gamma) = 0
    sol_k = k_star(beta, vartheta)
    sol_output = t.exp(sol_k/2) - 1
    return sol_output


def nu3(D, n, vartheta):
    # this function is used to obtain the solution of R(\gamma) = D/n * ln(2)
    '''
    @D:             the number of transmitting data bits
    @n:             finite blocklength
    @vartheta:      vartheta = Q^{-1}(spsilon) / sqrt(n)

    '''
    alpha = D/n * np.log(2)
    beta = np.exp(-alpha)
    sol_k = k_star(beta, vartheta)
    sol_output = np.exp(alpha + sol_k/2) - 1
    return sol_output


def R(gamma, vartheta):
    # this function is used to obtain the achievable rate
    '''
    @gamma:         the achivable sum rate, i.e., SINR. Its dimmension is (batchsize, K)
    '''
    one = t.ones_like(gamma)
    output = t.log(1+gamma) - vartheta * t.sqrt(1 - 1 / ((1+gamma) * (1+gamma)))
    return output


def tilde_gamma(P, H, sigmma):
    # this function is used to obtain tilde_gamma
    '''
    @P:             the maximum power constraint, its dimmension is (1)
    @H:             channel coefficient, its dimmension is (batchsize, Nt, K)
    @sigmma:        gaussian noise, suppose all users' sigmma is the same, its dimmesion is (1)
    '''
    HW_tmp = t.abs(H) * t.abs(H)                                                        # (batchsize, Nt, K)
    HW = t.sum(HW_tmp, 1)                                                               # (batchsize, K)
    output = P * HW / sigmma                                                            # (batchsize, K)
    return output


def bar_psi(x_k, x_k_1):
    # this function is used to compute bar_psi(psi_k)
    '''
    @x_k:           output of the k-th layer, its dimmension is (batchsize, 2*K*K+3K)
    @x_k_1:         output of the (k-1)-th layer, its dimmension is (batchsize, 2*K*K+3K)
    '''
    output = (1 - 2*x_k_1 + x_k) / ((1 - x_k_1) * (1 - x_k_1))                          # (batchsize, 2*K*K+3K)
    return output


def mu_psi(x_k, x_k_1):
    # this function is used to compute mu(psi_k)
    '''
    @x_k:           output of the k-th layer, its dimmension is (batchsize, 2*K*K+3K)
    @x_k_1:         output of the (k-1)-th layer, its dimmension is (batchsize, 2*K*K+3K)
    '''
    output = t.sqrt(x_k_1) / 2 + x_k / (2 * t.sqrt(x_k_1))                              # (batchsize, 2*K*K+3K)
    return output


def gradient_g(K, x_k, x_k_1, mu_k):
    # this function is used to compute the gradient of function g at x_k
    '''
    @K:             number of users
    @x_k:           the output of the k-th layer, (batchsize, 2*K*K+3*K)
    @x_k_1:         the output of the (k-1)-th layer, (batchsize, 2*K*K+3*K)
    @mu_k:          the barrier parameter outputted by the k-th layer
    '''
    Delta = indicator(K)             # indicator matrix, (K, K)
    # output = t.zeros_like(x_k)
    for i in range(K):
        m_i = 1 / ((1 - t.sum(Delta[2*K+i] * x_k_1, -1)) * (1 - t.sum(Delta[2*K+i] * x_k_1, -1)))   # (batchsize)      
        tmp_top = t.mm(m_i.unsqueeze(-1), Delta[2*K+i].unsqueeze(0)) - t.mm(2 * (1 + \
                    t.sum(Delta[3*K+i] * x_k, -1)).unsqueeze(-1), Delta[2*K+i].unsqueeze(0))        # (batchsize, 2*K*K+3*K)
        tmp_down = 2 * m_i - m_i * m_i + m_i * t.sum(Delta[2*K+i] * x_k, -1) - (1 +  \
                    t.sum(Delta[3*K+i] * x_k, -1)) * (1 + t.sum(Delta[3*K+i] * x_k, -1))            # (batch, )
        if i == 0: 
            output = tmp_top / tmp_down.unsqueeze(-1)                                               # (batchsize, 2*K*K+3*K)
        else:
            output = output +  tmp_top / tmp_down.unsqueeze(-1)                                     # (batchsize, 2*K*K+3*K)

    output = -mu_k * output
    return output


def prox_h1(x_k_t, alpha, Delta, lambda_k_t):
    # this function is used to compute the proximity operator of function h1 at x_k_t
    '''
    @x_k_t:          output of the t-th sub-iteration of the k-th layer, (batchsize, M, 2*K*K+3*K)
    @alpha:          prior weight of users, (2*K*K+3*K)
    @Delta:          the i-th indicator vector, (2*K*K+3*K, 2*K*K+3*K)
    @lambda_k_t:     stepsize of the t-th sub-iteration of the k-th layer,  (batchsize, )
    '''
    # prox_x = t.zeros_like(x_k_t)
    with t.no_grad():
        prox_x = x_k_t

    for i in range(len(x_k_t)):
        for j in range(len(x_k_t[0])):
            coe = (1 + t.sum(x_k_t[i, j] * Delta[j], -1)) + t.sqrt( t.pow(1+t.sum(x_k_t[i, j] \
                    * Delta[j], -1), 2) + 4*lambda_k_t[i]*alpha[j])
            coe = coe / 2                                                         # t.norm(Delta) is 1
            prox_x[i, j] = x_k_t[i, j] + coe * Delta[j]                           # (batchsize, 2*K*K+3*K, 2*K*K+3*K)
    return prox_x


def prox_h2(x_k_t, vartheta, alpha, Delta, lambda_k_t):
    # this function is used to compute the proximity operator of function h2 at x_k_t
    '''
    @x_k_t:          output of the t-th sub-iteration of the k-th layer, (batchsize, M, 2*K*K+3*K)
    @vartheta:       constant parameter
    @alpha:          prior weight of users, (2*K*K+3*K, )
    @Delta:          the i-th indicator vector, (2*K*K+3*K, 2*K*K+3*K)
    @lambda_k_t:     stepsize of the t-th sub-iteration of the k-th layer,  (batchsize,)
    '''
    # prox_x = t.zeros_like(x_k_t)                                # (batchsize, 2*K*K+3*K, 2*K*K+3*K)
    with t.no_grad():
        prox_x = x_k_t
    for i in range(len(x_k_t)):
        for j in range(len(x_k_t[0])):
            prox_x[i, j] = x_k_t[i, j] - lambda_k_t[i] * vartheta * alpha[j] * Delta[j]
    return prox_x


def prox_h3(x_k_t, A, b, lambda_k_t, mu_k):
    # this function is used to compute the proximity operator of function h3 at x_k_t
    '''
    @x_k_t:          output of the t-th sub-iteration of the k-th layer, (batchsize, M, 2*K*K+3*K), where M is the 
                     number of affine constraints
    @A:              left coefficient of a affine constraint, (batchsize, M, 2*K*K+3*K)
    @b:              right coefficient of a affine constraint, (batchsize, M)
    @lambda_k_t:     stepsize of the t-th sub-iteration of the k-th layer,  (batchsize, )
    @mu_k:           barrier parameter of the k-th layer
    '''
    tmp = b - t.sum(A * x_k_t, -1)                                                      # (batchsize, M)
    row_norm_A = t.sum(A * A, -1)                                                       # (batchsize, M)
    coe = tmp + t.sqrt(tmp * tmp + 4 * lambda_k_t * mu_k * row_norm_A)                  # (batchsize, M)
    coe = coe / (2 * row_norm_A)                                                        # (batchsize, M)
    prox_x = x_k_t + coe.unsqueeze(-1) * A                                              # (batchsize, M, 2*K*K+3*K)

    return prox_x


def prox_h4(x_k_t):
    # this function is used to compute the proximity operator of the function h4 at x_k_t
    '''
    @x_k_t:     output of the t-th sub-ieration of the k-th layer, (batchsize, M, 2*K*K+3*K)
    '''
    return x_k_t


def generate_layout(d0, Cell_radius, K):
    # this function is used to generate Cell network topology
    '''
    @d0:                        Reference distance d0
    @Cell_radius:               Cell radius
    @K:                         number of users
    '''
    BS_x = Cell_radius; BS_y = Cell_radius                                              #  coordinate of BS
    user_xs = np.zeros((K+1, 1)); user_ys = np.zeros((K+1, 1))                          #  coordinate of all users
    distance = []                                                                       #  distance between BS and users
    user_xs[0] = BS_x; user_ys[0] = BS_y
    for i in range(K):
        pair_distance = np.random.uniform(low = d0, high = Cell_radius)
        pair_angles = np.random.uniform(low = 0, high = np.pi * 2)
        user_x = BS_x + pair_distance * np.cos(pair_angles)
        user_y = BS_y + pair_distance * np.sin(pair_angles)
        user_xs[i+1] = user_x
        user_ys[i+1] = user_y
        distance.append(pair_distance)
    
    layout = np.concatenate((user_xs, user_ys), axis = 1)

    return layout, distance


def generate_h(num_h, K, Nt, d0, Cell_radius):
    #  this function is used to generate CSI
    '''
    @num_h:                 number of samples
    @K:                     number of users
    @Nt:                    number of transimite antennas
    @d0:                    Reference distance d0
    @Cell_radius:           Cell radius
    '''
    dst = np.zeros((num_h, K))
    layout = np.zeros((num_h, K+1, 2))
    for loop in range(num_h):
        layout_sample, dst_sample = generate_layout(d0, Cell_radius, K)
        layout[loop] = layout_sample
        dst[loop] = dst_sample
    
    # H = np.zeros((num_h, K, Nt))                                              # (num_h, K, Nt)
    tilde_H = (np.random.randn(num_h, K, Nt) +  1j * np.random.randn(num_h, K, Nt)) / np.sqrt(2)
    rho = 1 / (1 + np.power(dst / d0, 3))
    H = np.sqrt(rho).reshape(len(rho), len(rho[0]), 1) * tilde_H
    # note that the dtype of H is 'complex128', which should be transformed to 'complex64'
    return H.astype('complex64'), dst, layout


def PHI(H, W0, p0, sigmma):
    # this function is used to generate matrix Phi, which is used to implement initialization operations
    '''
    @H:                CSI, (batchsize, K, Nt)
    @W0:               beamforming vector, (batchsize, Nt, K)
    @p0:               initial power, (batchsize, K)
    @sigmma:           gaussian noise
    '''
    bar_gamma = gamma(p0, H, W0, sigmma)                                #  (batchsize, K)
    # H = H.conj()                                                            #  obtain the conjugate of H
    HW = t.abs(t.bmm(H, W0))                                             #  (batchsize, K, K)
    HW_2 = t.pow(HW, 2)
    eyes = t.empty_like(HW)                                                 #  used to remove non-diag items of HW
    for i in range(len(HW)):
        eyes[i] = t.eye(len(HW[0]), len(HW[0]))
    HW_negative = -HW_2
    HW_diag = HW_2 * eyes / bar_gamma.unsqueeze(-2)
    HW_negative = HW_negative * (1 - eyes)
    Phi = HW_negative + HW_diag                                             # (batchsize, K, K)
    return Phi


def tilde_q(Phi):
    # this function is used to compute tilde_q
    '''
    @Phi:                   matrix Phi, which is used to generate tilde_q, (batchsize, K, K)
    '''
    inv_Phi = np.linalg.pinv(Phi.numpy())
    # inv_Phi = t.inverse(Phi)                                                # obtain the inverse of Phi
    tilde_q_output = t.sum(t.from_numpy(inv_Phi), -1)                                     # (batchsize, K)
    return tilde_q_output


def tilde_V(phi):
    # this function is used to compute tilde_V(phi)
    '''
    @phi:                   variable phi, (batchsize, K)
    '''
    phi_tmp = (1 + phi) * (1 + phi)
    output = 1 - 1 / phi_tmp
    return output


def W_star(H, q, sigmma):
    # this function is used to compute w_star
    '''
    @H:                     CSI, (batchsize, K, Nt)
    @q:                     power of all users, (batchsize, K)
    @sigmma:                gaussian noise 
    '''
    bar_H = H / sigmma                                                                      #  (batchsize, K, Nt)
    # bar_H_conj = bar_H.conj()                                                             #  (batchsize, K, Nt)
    bar_H_conj = bar_H
    bar_H_usq = bar_H.unsqueeze(-1)                                                         #  (batchsize, K, Nt, 1)
    bar_H_conj = bar_H_conj.unsqueeze(-2)                                                   #  (batchsize, K, 1, Nt)
    HH = t.matmul(bar_H_usq, bar_H_conj)                                                    #  (batchsize, K, Nt, Nt)
    HH_q = HH * q.unsqueeze(-1).unsqueeze(-1)                                               #  (batchsize, K, Nt, Nt)
    HH_sum = t.sum(HH, dim = -3)                                                            #  (batchsize, Nt, Nt)
    for i in range(len(HH_sum[0,0])):
        HH_sum[:, i, i] = HH_sum[:, i, i] + 1
    coe_matrix = HH_sum                                                                     #  (batchsize, Nt, Nt)
    coe_matrix_inv = t.inverse(coe_matrix)                                                  #  obtain the inverse of coe_matrix
    bar_H_T = bar_H.permute(0, 2, 1)                                                        #  (batchsize, Nt, K)
    W_star_top = t.bmm(coe_matrix_inv, bar_H_T)                                             #  (batchsize, Nt, K)
    W_star_down = t.sqrt(t.sum(t.abs(W_star_top) * t.abs(W_star_top), 1)).unsqueeze(-2)     #  (batchsize, 1, K)
    W_star_output = W_star_top / W_star_down                                                #  (batchsize, Nt, K)
    
    return W_star_output


##  test the feasible of functions
if __name__ == "__main__":
    H = t.randn(3, 4) + 1j * t.randn(3, 4)
    W0 = t.randn(2, 4, 3) + 1j * t.randn(2, 4, 3)
    W = t.randn(2, 4, 3) + 1j * t.randn(2, 4, 3)
    p0 = t.randn(2, 3)
    sigmma = 1
    p = t.randn(2, 3)
    D = 256
    n = 128
    vartheta = 0.377
    x = t.abs(t.randn(2, 27))
    K = 3
    P = 40
    nu3_val = nu3(D, n, vartheta)
    print(nu3_val)