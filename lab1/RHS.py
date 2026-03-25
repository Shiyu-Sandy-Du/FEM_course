class RHS:
    def __init__(self, nu2, Xh, BC_dirichlet):
        self.nu2 = nu2
        self.Xh = Xh
        self.BC_dirichlet = BC_dirichlet
        
    def RHS_maker(self, u, t): # t is a dummy argument for the moment
        # impose the BC 
        # since the input in the RK substeps is not the field anymore
        u = self.dirichlet_BC_apply(u) 
        dudt = -self.Xh.dfluxdx_weak_compute_1d(u, self.nu2)\
               /self.Xh.mesh["B"]
        # impose the BC at the end of each RK substep
        u = self.dirichlet_BC_apply(u)
        return dudt
    
    def dirichlet_BC_apply(self, u): # apply a Dirichlet BC
        # Dirichlet BC
        u[0,0] = self.BC_dirichlet[0]
        u[-1,-1] = self.BC_dirichlet[-1]
        return u