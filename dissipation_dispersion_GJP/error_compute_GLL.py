import numpy as np
import matplotlib.pyplot as plt
from gll_lib import GLL_NodesAndWeights, GLL_DiffMatrix


def build_local_CG_matrices(p):
    n = p + 1
    xi, w = GLL_NodesAndWeights(n)
    dphi = GLL_DiffMatrix(xi)

    M = np.diag(w).astype(np.complex128)

    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            A[i, j] = w[i] * dphi[i, j]

    return M, A

def build_GJP_delta_matrices_Nodal(p, tau):
    """
    Four GJP endpoint-derivative matrices from Eq. (10) of Moura et al.
    With phi' vector transposed notated
    Returns
    -------
    Delta_pm : Δ_{+-} = (phi'_+) (phi'_-)^T
    Delta_pp : Δ_{++} = (phi'_+) (phi'_+)^T
    Delta_mm : Δ_{--} = (phi'_-) (phi'_-)^T
    Delta_mp : Δ_{-+} = (phi'_-) (phi'_+)^T

    Each matrix is local full size (p+1, p+1).
    """

    # endpoint locations: left ⊖, right ⊕
    n = p + 1
    xi, _ = GLL_NodesAndWeights(n)
    dphi = GLL_DiffMatrix(xi)

    dphi_m = dphi[0, :]   # phi'_- at xi = -1
    dphi_p = dphi[-1, :]   # phi'_+ at xi = +1

    Delta_pm = np.outer(dphi_p, dphi_m).astype(np.complex128)
    Delta_pp = np.outer(dphi_p, dphi_p).astype(np.complex128)
    Delta_mm = np.outer(dphi_m, dphi_m).astype(np.complex128)
    Delta_mp = np.outer(dphi_m, dphi_p).astype(np.complex128)
    
    Delta_eplus1  =   4.0 * tau * Delta_pm
    Delta_e       = - 4.0 * tau *(Delta_pp + Delta_mm)
    Delta_eminus1 =   4.0 * tau * Delta_mp
    
    return Delta_eminus1, Delta_e, Delta_eplus1


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


def build_Z_Nodal_CG_inviscid(p, kh, tau):
    M, A = build_local_CG_matrices(p)
    Delta_eminus1, Delta_e, Delta_eplus1 = build_GJP_delta_matrices_Nodal(p, tau)

    X = -A

    LM, CM, RM = split_CLR(M, p)
    LX, CX, RX = split_CLR(X, p)
    LDelta_eminus1, CDelta_eminus1, RDelta_eminus1 = split_CLR(Delta_eminus1, p)
    LDelta_e, CDelta_e, RDelta_e = split_CLR(Delta_e, p)
    LDelta_eplus1, CDelta_eplus1, RDelta_eplus1 = split_CLR(Delta_eplus1, p)


    e_minus = np.exp(-1j * kh)
    e_plus = np.exp(1j * kh)
    e_minusminus = np.exp(-2j * kh)
    e_plusplus = np.exp(2j * kh)

    Mhat = LM * e_minus + CM + RM * e_plus
    Xhat = LX * e_minus + CX + RX * e_plus
    Deltahat = LDelta_eminus1 * e_minusminus + CDelta_eminus1 * e_minus + RDelta_eminus1 \
             + LDelta_e       * e_minus      + CDelta_e                 + RDelta_e * e_plus \
             + LDelta_eplus1                 + CDelta_eplus1  * e_plus  + RDelta_eplus1 * e_plusplus

    Z = 2.0 * np.linalg.solve(Mhat, Xhat + Deltahat)

    return Z


def eigenvalues_Nodal_CG_inviscid(p, kh, tau):
    return np.linalg.eigvals(build_Z_Nodal_CG_inviscid(p, kh, tau))


def track_branches_Nodal_CG_inviscid(p, kh_vals, tau):
    kh_vals = np.asarray(kh_vals)
    branches = np.zeros((p, len(kh_vals)), dtype=np.complex128)

    eigs0 = eigenvalues_Nodal_CG_inviscid(p, kh_vals[0], tau)
    branches[:, 0] = eigs0[np.argsort(eigs0.imag)]

    for i in range(1, len(kh_vals)):
        eigs = list(eigenvalues_Nodal_CG_inviscid(p, kh_vals[i], tau))

        for b in range(p):
            prev = branches[b, i - 1]
            idx = np.argmin([abs(e - prev) for e in eigs])
            branches[b, i] = eigs.pop(idx)

    return branches


def physical_branch_Nodal_by_minimal_dispersion(p, kh_vals, tau):
    lam = np.zeros(len(kh_vals), dtype=np.complex128)

    for i, kh in enumerate(kh_vals):
        eigs = eigenvalues_Nodal_CG_inviscid(p, kh, tau)
        exact = -1j * kh
        idx = np.argmin(np.abs(eigs.imag - exact.imag))
        lam[i] = eigs[idx]

    return lam

def physical_branch_Nodal_by_continuation(p, kh_vals, tau):
    kh_vals = np.asarray(kh_vals)
    lam = np.zeros(len(kh_vals), dtype=np.complex128)

    # start near kh = 0
    i0 = np.argmin(np.abs(kh_vals))
    eigs0 = eigenvalues_Nodal_CG_inviscid(p, kh_vals[i0], tau)

    # at kh≈0, physical mode is the one closest to zero / exact
    lam[i0] = eigs0[np.argmin(np.abs(eigs0 + 1j * kh_vals[i0]))]

    # track to positive kh
    for i in range(i0 + 1, len(kh_vals)):
        eigs = eigenvalues_Nodal_CG_inviscid(p, kh_vals[i], tau)
        lam[i] = eigs[np.argmin(np.abs(eigs - lam[i - 1]))]

    # track to negative kh
    for i in range(i0 - 1, -1, -1):
        eigs = eigenvalues_Nodal_CG_inviscid(p, kh_vals[i], tau)
        lam[i] = eigs[np.argmin(np.abs(eigs - lam[i + 1]))]

    return lam


if __name__ == "__main__":
    p = 2
    npts = 50
    tau = 0.8 * (p+1)**(-4)
    # tau = 2 * 10**(-2)

    kh_vals = np.linspace(1e-6, np.pi * p, npts)
    xplot = kh_vals / p

    # load reference
    ref_data = np.loadtxt("ref_data/dispersion_temporal/P2.csv", delimiter=",")
    dispersion_temporal_p2_khp = ref_data[:, 0]
    dispersion_temporal_p2_omegahp = ref_data[:, 1]
    ref_data = np.loadtxt("ref_data/dissipation_temporal/P2.csv", delimiter=",")
    dissipation_temporal_p2_khp = ref_data[:, 0]
    dissipation_temporal_p2_omegahp = ref_data[:, 1]

    branches = track_branches_Nodal_CG_inviscid(p, kh_vals, tau)
    lam_phys = physical_branch_Nodal_by_continuation(p, kh_vals, tau)

    plt.figure(figsize=(4, 4))
    plt.plot(xplot, xplot, "--", color="gray")
    # for b in range(p):
    #     plt.plot(xplot, -branches[b].imag / p, color="tab:blue")
    plt.plot(xplot, -lam_phys.imag / p, color="tab:orange", linewidth=2, label="CG+GJP")
    plt.plot(dispersion_temporal_p2_khp, dispersion_temporal_p2_omegahp, \
             color="blue", linestyle="--", linewidth=2, label="Moura et al. 2021, modal")
    plt.xlabel(r"$kh/P$")
    plt.ylabel(r"$-\mathrm{Im}(\lambda h)/P$")
    plt.xlim(0, np.pi)
    plt.ylim(0, np.pi)
    plt.title(f"Nodal inviscid CG dispersion, P={p}")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/dispersion_nodal.png", dpi=300)

    plt.figure(figsize=(4, 4))
    plt.plot(xplot, np.zeros_like(xplot), "--", color="gray")
    # for b in range(p):
    #     plt.plot(xplot, branches[b].real / p, color="tab:blue")
    plt.plot(xplot, lam_phys.real / p, color="tab:orange", linewidth=2, label="CG+GJP")
    plt.plot(dissipation_temporal_p2_khp, dissipation_temporal_p2_omegahp, \
             color="blue", linestyle="--", linewidth=2, label="Moura et al. 2021, modal")
    plt.xlabel(r"$kh/P$")
    plt.ylabel(r"$\mathrm{Re}(\lambda h)/P$")
    plt.xlim(0, np.pi)
    plt.ylim(-4.0, 0.0)
    plt.title(f"Nodal inviscid CG dissipation, P={p}")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/dissipation_nodal.png", dpi=300)