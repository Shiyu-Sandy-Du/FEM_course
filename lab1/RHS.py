class RHS:
    def __init__(self, nu2, Xh):
        self.nu2 = nu2
        self.Xh = Xh
        
    def RHS_maker(self, u, t): # t is a dummy argument for the moment
        dudt = -self.Xh.dfluxdx_weak_compute_1d(u, self.nu2)\
               /self.Xh.mesh["B"]
        # dudt = self.Xh.avgC0(dudt)
        # dudt[0,0] = 1.0
        # dudt[-1,-1] = 0.0
        return dudt