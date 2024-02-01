import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
# import jax
# import jax.numpy as jnp

def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix

    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c**2, c * s], [c * s, s**2]])
    k1 = np.hstack([k0, -k0])
    K = E * A / L * np.vstack([k1, -k1])

    # stress matrix
    S = E / L * np.array([[-c, -s, c, s]])

    return K, S


def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices

    """

    idx = np.array([], dtype=int)

    for i in range(len(node)):

        n = node[i]
        start = DOF * (n - 1)
        finish = DOF * n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=int)))

    return idx

def truss1(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid):
    """Computes mass and stress for an arbitrary truss structure

    Parameters
    ----------
    nodes1 : ndarray of length nbar
        indices of the first nodes for bars. `nodes1` and `nodes2` can be in any order as long as consistent with phi
    nodes2 : ndarray of length nbar
        indices of the other nodes for bars
    phi : ndarray of length nbar (radians)
        defines orientation or bar
    A : ndarray of length nbar
        cross-sectional areas of each bar
    L : ndarray of length nbar
        length of each bar
    E : ndarray of length nbar
        modulus of elasticity of each bar
    rho : ndarray of length nbar
        material density of each bar
    Fx : ndarray of length nnode
        external force in the x-direction at each node
    Fy : ndarray of length nnode
        external force in the y-direction at each node
    rigid : list(boolean) of length nnode
        True if node_i is rigidly constrained

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress of each bar
    K : global stiffness 
        matrix (8x8)
    Ku : product (8x1)

    S : (10x8)


    """

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # mass
    mass = np.sum(rho * A * L)

    # stiffness and stress matrices
    K = np.zeros((DOF * n, DOF * n), dtype=complex)
    S = np.zeros((nbar, DOF * n), dtype=complex)
    # print("Initialized stiffness matrix {}".format(np.shape(K)))
    # print("Initialized S matrix {}".format(np.shape(S)))
    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        # print("Ksub size {} Ssub size".format(np.shape(Ksub)))
        # insert submatrix into global matrix
        idx = node2idx([nodes1[i], nodes2[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n * DOF, 1))

    for i in range(n):
        idx = node2idx([i + 1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx + 1, DOF)  # add 1 b.c. made indexing 1-based for convenience
    # print("pre-removal {}".format(K))
    # zeta =  np.linspace(0,12,12)
    # eta = np.linspace(0,12,12)
    # Zeta, Eta = np.meshgrid(zeta,eta)
    # plt.contourf(Zeta,Eta, K)
    # plt.title("Pre-removal")
    # plt.colorbar()
    # plt.show()
    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)
    # zeta =  np.linspace(0,8,8)
    # eta = np.linspace(0,8,8)
    # Zeta, Eta = np.meshgrid(zeta,eta)
    # plt.contourf(Zeta,Eta, K)
    # plt.title("Post-removal")
    # plt.colorbar()
    # plt.show()
    # solve for deflections
    # print("K global{} F global{} ".format(K,F)) 
    d = np.linalg.solve(K, F)
    # print("shape of S{}".format(np.shape(S)))

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress, K, np.dot(K,d), S, d


def truss(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid):
    """Computes mass and stress for an arbitrary truss structure

    Parameters
    ----------
    nodes1 : ndarray of length nbar
        indices of the first nodes for bars. `nodes1` and `nodes2` can be in any order as long as consistent with phi
    nodes2 : ndarray of length nbar
        indices of the other nodes for bars
    phi : ndarray of length nbar (radians)
        defines orientation or bar
    A : ndarray of length nbar
        cross-sectional areas of each bar
    L : ndarray of length nbar
        length of each bar
    E : ndarray of length nbar
        modulus of elasticity of each bar
    rho : ndarray of length nbar
        material density of each bar
    Fx : ndarray of length nnode
        external force in the x-direction at each node
    Fy : ndarray of length nnode
        external force in the y-direction at each node
    rigid : list(boolean) of length nnode
        True if node_i is rigidly constrained

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress of each bar

    """

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # mass
    mass = np.sum(rho * A * L)

    # stiffness and stress matrices
    K = np.zeros((DOF * n, DOF * n), dtype=complex)
    S = np.zeros((nbar, DOF * n), dtype=complex)
    # print("Initialized stiffness matrix {}".format(np.shape(K)))
    # print("Initialized S matrix {}".format(np.shape(S)))
    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        # print("Ksub size {} Ssub size".format(np.shape(Ksub)))
        # insert submatrix into global matrix
        idx = node2idx([nodes1[i], nodes2[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n * DOF, 1))

    for i in range(n):
        idx = node2idx([i + 1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx + 1, DOF)  # add 1 b.c. made indexing 1-based for convenience
    # print("pre-removal {}".format(K))
    # zeta =  np.linspace(0,12,12)
    # eta = np.linspace(0,12,12)
    # Zeta, Eta = np.meshgrid(zeta,eta)
    # plt.contourf(Zeta,Eta, K)
    # plt.title("Pre-removal")
    # plt.colorbar()
    # plt.show()
    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)
    # zeta =  np.linspace(0,8,8)
    # eta = np.linspace(0,8,8)
    # Zeta, Eta = np.meshgrid(zeta,eta)
    # plt.contourf(Zeta,Eta, K)
    # plt.title("Post-removal")
    # plt.colorbar()
    # plt.show()
    # solve for deflections
    # print("K global{} F global{} ".format(K,F)) 
    d = np.linalg.solve(K, F)
    # print("shape of S{}".format(np.shape(S)))

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress


def tenbartruss(A, grad_method='AJ', aggregate=False):
    """This is the subroutine for the 10-bar truss.
    TODO: You will need to complete it.

    Parameters
    ----------
    A : ndarray of length 10
        cross-sectional areas of all the bars
    grad_method : string (optional)
        gradient type.
        'FD' for finite difference,
        'CS' for complex step,
        'DT' for direct method,
        'AJ' for adjoint method,
        'AD' for automatic differentiation (extra credit).
    aggregate : bool (optional)
        If True, return the KS-aggregated stress constraint. If False, do not aggregate and return all stresses.
        The derivatives implementation for `aggreagate`=True is optional (extra credit).

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length 10 (if `aggregate`=False); float (if `aggregate`=True)
        stress of each bar or KS-aggregated stress value
    dmass_dA : ndarray of length 10
        derivative of mass w.r.t. each A
    dstress_dA : 10 x 10 ndarray (if `aggregate`=False); ndarray of length 10 if `aggregate`=True
        If `aggregated`=False, dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]
        If `aggregated`=True,  dstress_dA[j] is derivative of the KS-aggregated stress w.r.t. A[j]
    """

    # --- setup 10 bar truss ----
    # Truss node indexing:
    # wall > 1 ---------- 2 ---------- 3
    #          ++      ++ | ++      ++ |
    #            ++  ++   |   ++  ++   |
    #              ++     |     ++     |
    #            ++  ++   |   ++  ++   |
    #          ++      ++ | ++      ++ |
    # wall > 4 ---------- 5 ---------- 6

    # define bars by [node1, node2]
    bars_node = [
        [1, 2],   # bar 1
        [2, 3],   # bar 2 ...
        [4, 5],
        [5, 6],
        [2, 5],
        [3, 6],
        [1, 5],
        [2, 4],
        [2, 6],
        [3, 5]
    ]

    # arrays of the 1st and 2nd nodes
    nodes1 = []
    nodes2 = []
    for bar in bars_node:
        nodes1.append(bar[0])
        nodes2.append(bar[1])

    # bar orientations
    phi = np.deg2rad(np.array([0, 0, 0, 0, -90, -90, -45, -135, -45, -135]))

    # bar lengths
    bar_l = 10  # m
    ld = bar_l * np.sqrt(2)    # length of diagonal bars
    L  = np.array([bar_l, bar_l, bar_l, bar_l, bar_l, bar_l, ld, ld, ld, ld])
    
    # Young Modulus of each bar
    E = np.ones(10) * 70 * 10**9  # Pa
    
    # density of each bar
    rho = np.ones(10) * 2720  # kg/m^3
    
    # external loads
    P = 5 * 10**5     # N
    Fx = np.zeros(6)
    Fy = np.array([0, 0, 0, 0, -P, -P])
    
    # boundary condition (set True for clamped nodes)
    rigid = [True, False, False, True, False, False]

    # --- call truss function ----
    # This will compute the mass and stress of your truss structure
    mass, stress = truss(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid)
    # TODO: You may want to return additional variables from `truss` function for the implicit analytic methods.
    #       Feel free to modify `truss` function.

    # --- compute derivatives for provided grad_type ----
    # TODO: Implement derivatives for each method here
    if grad_method == 'FD' and aggregate==False:

        A0 =A.copy()
        # vectorized_functions = np.vectorize(lambda func: func(x0))
        # f0 = vectorized_functions(f)
        mass0, stress0 = truss(nodes1, nodes2, phi, A0, L, E, rho, Fx, Fy, rigid)
        # print("shape of stress {}".format(np.shape(stress0)))
        # f0 = func(x0) # if using def based function defination
        stress0  = np.array(abs(stress0))

        mass0  = np.array(mass0)
        # print("stress0 and mass0 shapes {}".format(np.shape(stress0)))
        nx = np.shape(A)[0]
        J1 = []
        J2 = []
        h = 10**(-8)
        for j in range(nx):
            dx = h*(1+ abs(A0[j]) ) #
            A[j] = A[j] + dx

            # function eval at perturbed point
            # Use np.vectorize to apply the vector of functions to the input value
            # vectorized_functions = np.vectorize(lambda func: func(x))
            mass_, stress_ = truss(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid)
            stress_  = np.array(abs(stress_))
            mass_ = np.array(mass_)
            
            # f_ = func(x) # if using def based function defination
            dmass_dA = (mass_ -mass0)/dx
            dstress_dA = (stress_ - stress0)/dx
            # print("the shape of dstress_dA {}".format(np.shape((dstress_dA))))
            J1.append(dmass_dA)
            J2.append(dstress_dA)
            print("J2 {}".format(np.shape(J2)))
            A[j] = A[j] - dx

        

        
    elif grad_method == 'CS' and aggregate==False:
    #     # implement complex step
 
        A = A.astype(complex)
        A0 =A

        nx = np.shape(A)[0]
        J1 = []
        J2 = []
        h = 10**(-200)
        for i in range(nx):
            temp = A[i]
            # print(temp)
            A[i] = complex(temp,h)
            # print(complex(2,3))
            # vectorized_functions = np.vectorize(lambda func: func(x))
            # f_ = vectorized_functions(f)
            mass_, stress_ = truss(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid)
            for j in range(nx):
                if stress_[j].real <0:
                    stress_[j] = complex(-np.array(stress_[j]).real,-np.array(stress_[j]).imag) 
                else:
                    stress_[j] = complex(np.array(stress_[j]).real,np.array(stress_[j]).imag) 


            f_ = np.array(stress_)
            J2.append((f_.imag)/h)
            A[i] = complex(A[i],-h)

    elif grad_method== 'DT' and aggregate==False:
        A0 = A
        Ac = A.copy()
        Ac = Ac.astype(complex)
        #start with K at the base design
        mass, stress, K0, ku0, S, u0 = truss1(nodes1, nodes2, phi, A0, L, E, rho, Fx, Fy, rigid) 
        Su = np.dot(S,u0)
        S =  np.sign(Su)*S
        nx = np.shape(A)[0]
        #inside the loop
        h = 10**(-200)
        J1 = []
        dsig_dx = []
        for i in range(nx):
            temp = Ac[i]
            print(temp)
            Ac[i] = complex(temp,h)
            # dx = h*(1+abs(A[i]))
            # A[i] = A[i] + dx

            # perturb x_i component and obtain the product (K(x+ e_i h)u - K(:,i)u)/dx (8x1)
            _, _, K, ku_, _, u = truss1(nodes1, nodes2, phi, Ac, L, E, rho, Fx, Fy, rigid) 
            # dr_dx = (ku - ku0)/dx
            ku = np.array(np.dot(K,u0))
            dr_dx = ku.imag/h
            Ac[i] = complex(temp,-h)

            # print("shape of ku {} \n ku0 {} \n dr_dx {}".format(np.shape(ku),np.shape(ku0),np.shape(dr_dx))) 
            # Solve for Phi_i and append
            Phi_i = np.linalg.solve(K0,dr_dx)

            # matrix product to obtain dsigma_dx_i and append
            dsig_dxi = -np.dot(S,Phi_i)
            dflat = dsig_dxi.flatten().copy()

            dsig_dx.append(dflat)
            # print("shape of dsig_dx {}".format(np.shape(dsig_dx)))
            # A[i] = A[i] - dx
            J2 = np.array(dsig_dx)

    # elif grad_method== 'AJ':

        

    elif grad_method== 'DT' and aggregate==True:
        A0 = A
        Ac = A.copy()
        Ac = Ac.astype(complex)
        #start with K at the base design
        mass, stress, K0, ku0, S, u0 = truss1(nodes1, nodes2, phi, A0, L, E, rho, Fx, Fy, rigid) 
        # Su = np.dot(S,u0)
        # S =  np.sign(Su)*S

        # Calculations for KS aggregation
        rho = 1000

        # Calculate Su
        Su = np.dot(S, u0)

        # Calculate the absolute values of sigma components
        abs_sigma = np.abs(Su)

        # Calculate the terms in the denominator of g(|σ|)
        exp_rho_abs_sigma_minus_max = np.exp(rho * (abs_sigma - np.max(abs_sigma)))

        # print(rho * (abs_sigma - np.max(abs_sigma)))
        # Calculate g(|σ|)
        g_sigma = np.max(abs_sigma) + np.log(exp_rho_abs_sigma_minus_max.sum()) / rho

        # Calculate the derivative dg(|σ|)/du
        derivative_g_sigma_u = np.dot((np.sign(Su) * S).T, exp_rho_abs_sigma_minus_max) / (rho * exp_rho_abs_sigma_minus_max.sum())   



        nx = np.shape(A)[0]
        #inside the loop
        h = 10**(-200)
        J1 = []
        dsig_dx = []
        for i in range(nx):
            temp = Ac[i]
            print(temp)
            Ac[i] = complex(temp,h)
            # dx = h*(1+abs(A[i]))
            # A[i] = A[i] + dx

            # perturb x_i component and obtain the product (K(x+ e_i h)u - K(:,i)u)/dx (8x1)
            _, _, K, ku_, _, u = truss1(nodes1, nodes2, phi, Ac, L, E, rho, Fx, Fy, rigid) 
            # dr_dx = (ku - ku0)/dx
            ku = np.array(np.dot(K,u0))
            dr_dx = ku.imag/h
            Ac[i] = complex(temp,-h)

            # print("shape of ku {} \n ku0 {} \n dr_dx {}".format(np.shape(ku),np.shape(ku0),np.shape(dr_dx))) 
            # Solve for Phi_i and append
            Phi_i = np.linalg.solve(K0,dr_dx)

            # matrix product to obtain dsigma_dx_i and append
            dsig_dxi = -np.dot(derivative_g_sigma_u.T,Phi_i)
            dflat = dsig_dxi.flatten().copy()

            dsig_dx.append(dflat)
            # print("shape of dsig_dx {}".format(np.shape(dsig_dx)))
            # A[i] = A[i] - dx
            J2 = np.array(dsig_dx)    

        


        

    elif grad_method== 'AJ' and aggregate==False:
        # get the S, K matrix
    
        # matrix multiplication to get Psi_j

        # compute dK_dx by complex step and form (8x10) matrix

        # loop over the nt and get all dsig_i_dx
        A0 = A
        Ac = A.copy()
        Ac = Ac.astype(complex)
        #start with K at the base design
        mass, stress, K0, ku0, S, u0 = truss1(nodes1, nodes2, phi, A0, L, E, rho, Fx, Fy, rigid) 
        Su = np.dot(S,u0)
        S =  np.sign(Su)*S

        # Calculations for KS aggregation
            

        # dkdx
        nx = np.shape(A)[0]
        h = 10**(-200)
        dkdx = []
        for i in range(nx):
            temp = Ac[i]
            # print(temp)
            Ac[i] = complex(temp,h)
            
            _, _, K, ku_, _, u = truss1(nodes1, nodes2, phi, Ac, L, E, rho, Fx, Fy, rigid) 
            
            ku = np.array(np.dot(K,u0))
            dr_dx = ku.imag/h
            Ac[i] = complex(temp,-h)
            dflat = dr_dx.T.flatten().copy()
            dkdx.append(dflat)

        print((dkdx))
        nt = np.shape(stress)[0]
        J2 = []
        J1 = []
        print(np.shape(dkdx))
        dkdx = np.array(dkdx).T
        for j in range(nt):
            Sj = S[j,:].reshape((8,1)) #(8,)
            
            Psi_j = np.linalg.solve(K, Sj)
            print(np.shape(Psi_j))
            # column vector dsig_i_dx

            dsigi_dx = -np.dot(Psi_j.T, dkdx)
            print(np.shape(dsigi_dx))
            dr_dxflat = dsigi_dx.T.flatten().copy()
            J2.append(dr_dxflat)


    elif grad_method== 'AJ' and aggregate== True:
        # get the S, K matrix
    
        # matrix multiplication to get Psi_j

        # compute dK_dx by complex step and form (8x10) matrix

        # loop over the nt and get all dsig_i_dx
        A0 = A
        Ac = A.copy()
        Ac = Ac.astype(complex)
        #start with K at the base design
        mass, stress, K0, ku0, S, u0 = truss1(nodes1, nodes2, phi, A0, L, E, rho, Fx, Fy, rigid) 
        # Su = np.dot(S,u0)
        # S =  np.sign(Su)*S

        # Calculations for KS aggregation
        rho = 1000

        # Calculate Su
        Su = np.dot(S, u0)

        # Calculate the absolute values of sigma components
        abs_sigma = np.abs(Su)

        # Calculate the terms in the denominator of g(|σ|)
        exp_rho_abs_sigma_minus_max = np.exp(rho * (abs_sigma - np.max(abs_sigma)))

        # print(rho * (abs_sigma - np.max(abs_sigma)))
        # Calculate g(|σ|)
        g_sigma = np.max(abs_sigma) + np.log(exp_rho_abs_sigma_minus_max.sum()) / rho

        # Calculate the derivative dg(|σ|)/du
        derivative_g_sigma_u = np.dot((np.sign(Su) * S).T, exp_rho_abs_sigma_minus_max) / (rho * exp_rho_abs_sigma_minus_max.sum())   

        # dkdx
        nx = np.shape(A)[0]
        h = 10**(-200)
        dkdx = []
        for i in range(nx):
            temp = Ac[i]
            # print(temp)
            Ac[i] = complex(temp,h)
            
            _, _, K, ku_, _, u = truss1(nodes1, nodes2, phi, Ac, L, E, rho, Fx, Fy, rigid) 
            
            ku = np.array(np.dot(K,u0))
            dr_dx = ku.imag/h
            Ac[i] = complex(temp,-h)
            dflat = dr_dx.T.flatten().copy()
            dkdx.append(dflat)

        print((dkdx))
        nt = np.shape(stress)[0]
        J2 = []
        J1 = []
        print(np.shape(dkdx))
        dkdx = np.array(dkdx).T
        for j in range(1): # changes for KS
            # Sj = S[j,:].reshape((8,1)) #(8,)
            derivative_g_sigma_u
            Psi_j = np.linalg.solve(K, derivative_g_sigma_u) 
            print(np.shape(Psi_j))
            # column vector dsig_i_dx

            dsigi_dx = -np.dot(Psi_j.T, dkdx)
            print(np.shape(dsigi_dx))
            dr_dxflat = dsigi_dx.T.flatten().copy()
            J2.append(dr_dxflat)

        
    else:
        raise("FD and CS aren't programmed to work with aggregate")
        
    
    return mass, stress, J1, np.array(J2)

def tenbarJAX(A):
    """This is the subroutine for the 10-bar truss.
    TODO: You will need to complete it.

    Parameters
    ----------
    A : ndarray of length 10
        cross-sectional areas of all the bars
    grad_method : string (optional)
        gradient type.
        'FD' for finite difference,
        'CS' for complex step,
        'DT' for direct method,
        'AJ' for adjoint method,
        'AD' for automatic differentiation (extra credit).
    aggregate : bool (optional)
        If True, return the KS-aggregated stress constraint. If False, do not aggregate and return all stresses.
        The derivatives implementation for `aggreagate`=True is optional (extra credit).

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length 10 (if `aggregate`=False); float (if `aggregate`=True)
        stress of each bar or KS-aggregated stress value
    dmass_dA : ndarray of length 10
        derivative of mass w.r.t. each A
    dstress_dA : 10 x 10 ndarray (if `aggregate`=False); ndarray of length 10 if `aggregate`=True
        If `aggregated`=False, dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]
        If `aggregated`=True,  dstress_dA[j] is derivative of the KS-aggregated stress w.r.t. A[j]
    """

    # --- setup 10 bar truss ----
    # Truss node indexing:
    # wall > 1 ---------- 2 ---------- 3
    #          ++      ++ | ++      ++ |
    #            ++  ++   |   ++  ++   |
    #              ++     |     ++     |
    #            ++  ++   |   ++  ++   |
    #          ++      ++ | ++      ++ |
    # wall > 4 ---------- 5 ---------- 6

    # define bars by [node1, node2]
    bars_node = [
        [1, 2],   # bar 1
        [2, 3],   # bar 2 ...
        [4, 5],
        [5, 6],
        [2, 5],
        [3, 6],
        [1, 5],
        [2, 4],
        [2, 6],
        [3, 5]
    ]

    # arrays of the 1st and 2nd nodes
    nodes1 = []
    nodes2 = []
    for bar in bars_node:
        nodes1.append(bar[0])
        nodes2.append(bar[1])

    # bar orientations
    phi = np.deg2rad(np.array([0, 0, 0, 0, -90, -90, -45, -135, -45, -135]))

    # bar lengths
    bar_l = 10  # m
    ld = bar_l * np.sqrt(2)    # length of diagonal bars
    L  = np.array([bar_l, bar_l, bar_l, bar_l, bar_l, bar_l, ld, ld, ld, ld])
    
    # Young Modulus of each bar
    E = np.ones(10) * 70 * 10**9  # Pa
    
    # density of each bar
    rho = np.ones(10) * 2720  # kg/m^3
    
    # external loads
    P = 5 * 10**5     # N
    Fx = np.zeros(6)
    Fy = np.array([0, 0, 0, 0, -P, -P])
    
    # boundary condition (set True for clamped nodes)
    rigid = [True, False, False, True, False, False]

    # --- call truss function ----
    # This will compute the mass and stress of your truss structure
    mass, stress = truss(nodes1, nodes2, phi, A, L, E, rho, Fx, Fy, rigid)
    
        
    
    return  stress




if __name__ =="__main__":
    # import time

    # Record the start time
    # start_time = time.time()
    A = 0.01*np.ones((10,1))
    mass, stress, dmass_dA, dstress_dA1= tenbartruss(A, grad_method='FD', aggregate=False)

    # print("mass {} \n stress {} \n dmass {} \n dstress {} and \n the shape of dstress {}". format(mass, stress, dmass_dA, dstress_dA1, np.shape(dstress_dA1)))

    # mass, stress, dmass_dA, dstress_dA2= tenbartruss(A, grad_method='FD', aggregate=False)

    # print("mass {} \n stress {} \n dmass {} \n dstress {} and \n the shape of dstress {}". format(mass, stress, dmass_dA, dstress_dA2, np.shape(dstress_dA2)))
    # # Use jax.grad to get the derivative
    # grad_stress = jax.grad(tenbarJAX)
    
    # # Now you can compute the derivative with respect to A
    # A_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Replace with your initial values
    # derivative_wrt_A = grad_stress(A_values)

    # print("Derivative with respect to A:", derivative_wrt_A)
    # Record the end time
    # end_time = time.time()

    # Calculate the runtime
    # runtime = end_time - start_time

    # print(f"Runtime: {runtime} seconds")

    # max_error = np.max(np.abs(dstress_dA1 - dstress_dA2)/np.abs(dstress_dA1))

    # print("Maximum Error:", max_error)
    x =  np.linspace(0,10,10)
    # plt.plot(x,np.array(dstress_dA1).real.ravel())
    # plt.xlabel("design variables number")
    # plt.ylabel("Derivative")
    # plt.show()
    # y = np.linspace(0,10,10)
    # X, Y = np.meshgrid(x,y)
    # plt.contourf(X[:,:],Y[:,:], np.array(dstress_dA1)[:,:])
    # plt.colorbar()
    # plt.show()
