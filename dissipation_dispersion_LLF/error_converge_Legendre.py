import numpy as np
from gll_lib import (
    GL_NodesAndWeights,
    GL_DiffMatrix,
    lgP,
    lgPx_mat,
)

# ------------------------------------------------------------
# Build element-local matrices for 1D CG/SEM on [-1,1]
# with nodal GLL Lagrange basis
# ------------------------------------------------------------
def build_local_matrices(p):
    """
    Build A1, B1, IM1, A1T, B1T for polynomial order p.

    Assumptions
    -----------
    - Reference element is [-1, 1]
    - Basis functions are the nodal GLL Lagrange basis of degree p
    - There are p+1 GLL nodes
    - Integrals are evaluated with GLL quadrature
    """

    n = p + 1
    xi, w = GL_NodesAndWeights(n)     # n nodes => polynomial degree p
    D = GL_DiffMatrix(xi)           # D[i,j] = d l_j / dx evaluated at x_i (nodal derivative matrix)

    # Legendre modal basis sampled at quadrature nodes from gll_lib:
    # lgPx_mat returns PHI_N[i, m] = P_m(x_i), so transpose to [m, i].
    phi = lgPx_mat(p, xi).T
    dphi = phi @ D.T  # dphi[m, i] = sum_j phi[m, j] D[j, i] = P_m'(x_i)

    # --------------------------------------------------------
    # Basis values at the endpoints:
    # phi_k(1)   and   phi_m(-1)
    #
    # For Legendre polynomials P_m(x):
    # P_m(1) = 1 and P_m(-1) = (-1)^m
    # --------------------------------------------------------
    phi_at_plus1 = np.array([lgP(m, np.array([1.0]))[0] for m in range(n)], dtype=np.float64)
    phi_at_minus1 = np.array([lgP(m, np.array([-1.0]))[0] for m in range(n)], dtype=np.float64)

    # --------------------------------------------------------
    # A1[m,k] = ∫ phi_k(x) d/dx(phi_m(x)) dx - phi_k(1) phi_m(1)
    #
    # For modal Legendre basis sampled on quadrature nodes,
    # evaluate the volume term by quadrature.
    # --------------------------------------------------------
    A1 = np.zeros((n, n), dtype=np.complex128)
    for m in range(n):
        for k in range(n):
            volume_term = w.T @ (phi[k,:] * dphi[m, :])
            boundary_term = phi_at_plus1[k] * phi_at_plus1[m]
            A1[m, k] = volume_term - boundary_term

    # --------------------------------------------------------
    # B1[m,k] = phi_k(1) phi_m(-1)
    # --------------------------------------------------------
    B1 = np.zeros((n, n), dtype=np.complex128)
    for m in range(n):
        for k in range(n):
            B1[m, k] = phi_at_plus1[k] * phi_at_minus1[m]

    # --------------------------------------------------------
    # IM1[m,k] = (2*m-1) KroneckerDelta[m,k] in Mathematica
    # with m = 1,...,p+1
    #
    # In Python zero-based indexing:
    # diag = [1, 3, 5, ..., 2p+1]
    # --------------------------------------------------------
    IM1 = np.diag([(2*m+1)/2 for m in range(n)]).astype(np.complex128)
    A1T = IM1 @ A1
    B1T = IM1 @ B1

    return xi, w, D, A1, B1, IM1, A1T, B1T


# ------------------------------------------------------------
# Build G1(xi)
# ------------------------------------------------------------
def build_G1(p, xi):
    _, _, _, A1, B1, IM1, A1T, B1T = build_local_matrices(p)
    G1h = 2 * (A1T + np.exp(-1j * xi) * B1T)
    return G1h


# ------------------------------------------------------------
# Eigenvalues of G1 for one xi
# ------------------------------------------------------------
def eigenvalues_G1(p, xi):
    G1 = build_G1(p, xi)
    return np.linalg.eigvals(G1)


# ------------------------------------------------------------
# Track one eigenvalue branch continuously over xi values
# ------------------------------------------------------------
def track_eigenvalue_branch(p, xi_vals, branch='physical'):
    """
    Track one eigenvalue branch over xi_vals.

    branch options:
    - 'physical': choose the eigenvalue closest to 1j*xi at the first xi
    - integer j : choose j-th eigenvalue after sorting by imag part initially

    Returns
    -------
    lam_track : complex ndarray
        tracked eigenvalue branch
    """
    xi_vals = np.asarray(xi_vals)
    lam_track = np.zeros(len(xi_vals), dtype=np.complex128)

    eig0 = np.linalg.eigvals(build_G1(p, xi_vals[0]))

    if branch == 'physical':
        idx = np.argmin(np.abs(eig0 - 1j * xi_vals[0]))
    else:
        eig0_sorted = eig0[np.argsort(np.imag(eig0))]
        lam_track[0] = eig0_sorted[branch]
        prev = lam_track[0]

        for i in range(1, len(xi_vals)):
            eigs = np.linalg.eigvals(build_G1(p, xi_vals[i]))
            idx = np.argmin(np.abs(eigs - prev))
            lam_track[i] = eigs[idx]
            prev = lam_track[i]
        return lam_track

    lam_track[0] = eig0[idx]
    prev = lam_track[0]

    for i in range(1, len(xi_vals)):
        eigs = np.linalg.eigvals(build_G1(p, xi_vals[i]))
        idx = np.argmin(np.abs(eigs - prev))
        lam_track[i] = eigs[idx]
        prev = lam_track[i]

    return lam_track

# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    for p in [1,2]:
        # xi = 0.2

        # x, w, D, A1, B1, IM1, A1T, B1T = build_local_matrices(p)
        # G1 = build_G1(p, xi)
        # eigs = np.linalg.eigvals(G1)

        # np.set_printoptions(precision=6, suppress=True)

        # print("GLL nodes x =")
        # print(x)
        # print("\nGLL weights w =")
        # print(w)
        # print("\nDifferentiation matrix D =")
        # print(D)

        # print("\nA1 =")
        # print(A1)
        # print("\nB1 =")
        # print(B1)
        # print("\nIM1 =")
        # print(IM1)

        # print("\nG1(xi) =")
        # print(G1)

        # print("\nEigenvalues of G1 =")
        # print(eigs)

        # Track the physical branch over a range of xi
        xi_vals = np.linspace(1e-4, np.pi*p, 1000)
        lam_physical = track_eigenvalue_branch(p, xi_vals, branch="physical")
        lam_list = []
        for i in range(p+1):
            lam = track_eigenvalue_branch(p, xi_vals, branch=i)
            lam_list.append(lam)

        ##########################################################
        ### plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        # for i in range(p+1):
        #     lam = lam_list[i]
        #     plt.plot(np.linspace(0, np.pi, 100), -lam.imag/p, label=f'branch {i}', markersize=4)
        plt.plot(xi_vals/p, -lam_physical.imag/p, label=f'p={p}', color='red', markersize=4)
        plt.plot(xi_vals/p, xi_vals/p, linestyle='--', color='gray')
        plt.xlabel(r'$\xi$/P')
        plt.ylabel(r'$-Imag(\lambda h)/P$')
        plt.xlim(0, np.pi)
        plt.xticks(np.linspace(0, np.pi, 5), ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        plt.title(f'Imaginary part of eigenvalues')
        # plt.legend()
        plt.grid()
        plt.savefig(f'figures/tracked_branch_imag_Legendre_p{p}.png', dpi=300)

        plt.figure(figsize=(5, 5))
        # for i in range(p+1):
        #     lam = lam_list[i]
        #     plt.plot(np.linspace(0, np.pi, 100), lam.real/p, label=f'branch {i}', markersize=4)
        plt.plot(xi_vals/p, lam_physical.real/p, label=f'p={p}', color='red', markersize=4)
        plt.plot(xi_vals/p, [0 for _ in range(len(xi_vals))], linestyle='--', color='gray')
        plt.xlabel(r'$\xi$/P')
        plt.ylabel(r'$Real(\lambda h)$/P')
        plt.xlim(0, np.pi)
        plt.xticks(np.linspace(0, np.pi, 5), ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        plt.title(f'Real part of eigenvalues')
        # plt.legend()
        plt.grid()
        plt.savefig(f'figures/tracked_branch_real_Legendre_p{p}.png', dpi=300)
        
        plt.figure(figsize=(6, 5))
        plt.plot(np.log10(xi_vals), np.log10(abs(lam_physical.real)), label=f'p={p}', color='red', markersize=4)
        if p == 1:
            plt.plot(np.log10(xi_vals), (2*p + 2) * np.log10(xi_vals) - 3, linestyle='--', color='gray')
        elif p == 2:
            plt.plot(np.log10(xi_vals), (2*p + 2) * np.log10(xi_vals) - 5, linestyle='--', color='gray')
        plt.xlabel(r'$log_{10}(\xi$)')
        plt.ylabel(r'$log_{10}(abs(Real(\lambda h)))$)')
        plt.xlim(np.log10(xi_vals[0]),np.log10(xi_vals[-1]))
        plt.ylim(-17,0.5)
        plt.title(f'Dispersion Error')
        # plt.legend()
        plt.grid()
        plt.savefig(f'figures/dissipation_error_Legendre_p{p}.png', dpi=300)
        
        plt.figure(figsize=(6, 5))
        plt.plot(np.log10(xi_vals), np.log10(abs(-lam_physical.imag - xi_vals)), label=f'p={p}', color='red', markersize=4)
        if p == 1:
            plt.plot(np.log10(xi_vals), (2*p + 3) * np.log10(xi_vals) - 3, linestyle='--', color='gray')
        elif p == 2:
            plt.plot(np.log10(xi_vals), (2*p + 3) * np.log10(xi_vals) - 5, linestyle='--', color='gray')
        plt.xlabel(r'$log_{10}(\xi$)')
        plt.ylabel(r'$log_{10}(abs(Imag(\lambda h)))$)')
        plt.xlim(np.log10(xi_vals[0]),np.log10(xi_vals[-1]))
        plt.ylim(-17,0.5)
        plt.title(f'Dispersion Error')
        # plt.legend()
        plt.grid()
        plt.savefig(f'figures/dispersion_error_Legendre_p{p}.png', dpi=300)