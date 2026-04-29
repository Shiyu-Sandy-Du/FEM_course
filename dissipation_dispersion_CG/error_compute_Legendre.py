import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from gll_lib import GL_NodesAndWeights


def modal_basis_and_derivative(p, x):
    """
    Hierarchical/modal CG basis on [-1,1].

    Ordering:
        0       : left vertex mode
        1:p     : interior bubble modes
        p       : right vertex mode
    """
    x = np.asarray(x)
    n = p + 1

    phi = np.zeros((len(x), n))
    dphi = np.zeros((len(x), n))

    # left vertex mode
    phi[:, 0] = 0.5 * (1.0 - x)
    dphi[:, 0] = -0.5

    # interior bubble modes
    for j in range(1, p):
        L_plus = Legendre.basis(j + 1)
        L_minus = Legendre.basis(j - 1)

        phi[:, j] = L_plus(x) - L_minus(x)
        dphi[:, j] = L_plus.deriv()(x) - L_minus.deriv()(x)

    # right vertex mode
    phi[:, p] = 0.5 * (1.0 + x)
    dphi[:, p] = 0.5

    return phi, dphi


def build_local_modal_CG_matrices(p):
    """
    Inviscid modal CG local matrices.

    M_ij = ∫ phi_i phi_j dξ
    A_ij = ∫ phi_i dphi_j/dξ dξ
    """
    nq = max(2 * p + 3, 20)
    xq, wq = GL_NodesAndWeights(nq)

    phi, dphi = modal_basis_and_derivative(p, xq)

    W = np.diag(wq)

    M = phi.T @ W @ phi
    A = phi.T @ W @ dphi

    return M.astype(np.complex128), A.astype(np.complex128)


def split_CLR(B, p):
    """
    CG block split.

    Local ordering:
        [left vertex, bubbles..., right vertex]

    Global unknowns per element:
        [left vertex, bubbles...]

    The right vertex is the left vertex of the next element.
    """
    C = B[:p, :p].copy()

    L = np.zeros((p, p), dtype=np.complex128)
    R = np.zeros((p, p), dtype=np.complex128)

    L[0, :] = B[p, :p]
    R[:, 0] = B[:p, p]

    C[0, 0] += B[p, p]

    return L, C, R


def build_Z_modal_CG_inviscid(p, kh):
    M, A = build_local_modal_CG_matrices(p)

    X = -A

    LM, CM, RM = split_CLR(M, p)
    LX, CX, RX = split_CLR(X, p)

    e_minus = np.exp(-1j * kh)
    e_plus = np.exp(1j * kh)

    Mhat = LM * e_minus + CM + RM * e_plus
    Xhat = LX * e_minus + CX + RX * e_plus

    Z = 2.0 * np.linalg.solve(Mhat, Xhat)

    return Z


def eigenvalues_modal_CG_inviscid(p, kh):
    return np.linalg.eigvals(build_Z_modal_CG_inviscid(p, kh))


def track_branches_modal_CG_inviscid(p, kh_vals):
    kh_vals = np.asarray(kh_vals)
    branches = np.zeros((p, len(kh_vals)), dtype=np.complex128)

    eigs0 = eigenvalues_modal_CG_inviscid(p, kh_vals[0])
    branches[:, 0] = eigs0[np.argsort(eigs0.imag)]

    for i in range(1, len(kh_vals)):
        eigs = list(eigenvalues_modal_CG_inviscid(p, kh_vals[i]))

        for b in range(p):
            prev = branches[b, i - 1]
            idx = np.argmin([abs(e - prev) for e in eigs])
            branches[b, i] = eigs.pop(idx)

    return branches


def physical_branch_modal_CG_inviscid(p, kh_vals):
    lam = np.zeros(len(kh_vals), dtype=np.complex128)

    for i, kh in enumerate(kh_vals):
        eigs = eigenvalues_modal_CG_inviscid(p, kh)
        exact = -1j * kh
        idx = np.argmin(np.abs(eigs - exact))
        lam[i] = eigs[idx]

    return lam


if __name__ == "__main__":
    p = 6
    npts = 300

    kh_vals = np.linspace(-np.pi * p, np.pi * p, npts)
    xplot = kh_vals / p

    branches = track_branches_modal_CG_inviscid(p, kh_vals)
    lam_phys = physical_branch_modal_CG_inviscid(p, kh_vals)

    plt.figure(figsize=(4, 4))
    plt.plot(xplot, xplot, "--", color="gray")
    for b in range(p):
        plt.plot(xplot, -branches[b].imag / p, color="tab:blue")
    plt.plot(xplot, -lam_phys.imag / p, color="tab:orange", linewidth=2)
    plt.xlabel(r"$kh/P$")
    plt.ylabel(r"$-\mathrm{Im}(\lambda)/P$")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.title(f"Modal inviscid CG dispersion, P={p}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/tracked_branch_imag_modal_CG_inviscid.png", dpi=300)

    plt.figure(figsize=(4, 4))
    plt.plot(xplot, np.zeros_like(xplot), "--", color="gray")
    for b in range(p):
        plt.plot(xplot, branches[b].real / p, color="tab:blue")
    plt.plot(xplot, lam_phys.real / p, color="tab:orange", linewidth=2)
    plt.xlabel(r"$kh/P$")
    plt.ylabel(r"$\mathrm{Re}(\lambda)/P$")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.title(f"Modal inviscid CG dissipation, P={p}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/tracked_branch_real_modal_CG_inviscid.png", dpi=300)