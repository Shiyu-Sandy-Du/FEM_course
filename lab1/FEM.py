import numpy as np
from gll_lib import gLLNodesAndWeights, gll_diff_matrix

class FEM_space:
    ## Input parameters to initialise a finite element space
    # node_type: type of internal nodes, e.g. "GLL"
    # P: polynomial order
    # element: a 1D array for the element discretisation of the domain
    # dim: dimension of the problem
    # mapping_method: the mapping method to reference coordinates
    def __init__(self, node_type, P, element, dim, mapping_method):
        self.node_type = node_type
        self.P = P # polynomial order
        self.lx = P+1 # number of points per element
        self.dim = dim # dimension of the problem
        self.nelem = len(element) - 1
        if dim != 1:
            raise ValueError("Xh is only implemented for 1D now")

        if self.node_type == "GLL":
            ###### Build up local reference coordinates
            self.xi, self.weights = gLLNodesAndWeights(self.lx)
            self.D = gll_diff_matrix(self.xi)
            self.Dt = self.D.T

            ###### Build up the FEM mesh
            self.mesh = {}
            self.mesh["h"] = np.diff(element)
            # the mesh
            self.mesh["x"] = np.empty((self.nelem, self.lx))
            # Jacobian
            self.mesh["J"] = np.empty((self.nelem, self.lx))
            # the inverse of Jacobian
            self.mesh["Jinv"] = np.empty((self.nelem, self.lx))
            # the mass matrix
            self.mesh["B"] = np.empty((self.nelem, self.lx))
            # set up the Jacobian and its inverse
            if mapping_method == "linear":
                for i in range(self.nelem):
                    self.mesh["Jinv"][i,:] = 2.0/self.mesh["h"][i] 
                    self.mesh["J"][i,:] = self.mesh["h"][i]/2.0
                    self.mesh["B"][i,:] = self.weights * self.mesh["h"][i]/2.0
                self.mesh["B"] = self.gs_add(self.mesh["B"])
            else:
                raise ValueError("the mapping method is only" + \
                      " implemented for linear mapping")
            # set up the mesh
            for i in range(self.nelem):
                self.mesh["x"][i,:] = element[i] + \
                                      self.mesh["J"][i] * (self.xi + 1.0)

        else:
            raise ValueError("unknown node_type:" + self.node_type + "\n" + \
                             "please choose GLL as the input")

    # perform elementwise maxtrix multiplication
    # A -- (lx,lx)
    # u -- (nelm, lx)
    # the output is of the same shape as u
    def conv(self, A, u):
        return u @ A.T

    # Force a field u with C0 continuity in the end by averaging
    def gs_avgC0(self, u_discontinuous):
        # Gather
        u_left = u_discontinuous[1:,0]
        u_right = u_discontinuous[:-1,-1]
        u_interface = (u_left + u_right)/2.0
        # Scatter
        u_discontinuous[1:,0] = u_interface
        u_discontinuous[:-1,-1] = u_interface
        return u_discontinuous
    
    def gs_add(self, u_discontinuous):
        # Gather
        u_left = u_discontinuous[1:,0]
        u_right = u_discontinuous[:-1,-1]
        u_interface = u_left + u_right
        # Scatter
        u_discontinuous[1:,0] = u_interface
        u_discontinuous[:-1,-1] = u_interface
        return u_discontinuous

    # Compute the flux function on element basis
    def dfluxdx_weak_compute_1d(self, u, nu2, BC_dirichlet):
        u[0,0] = BC_dirichlet[0]
        u[-1,-1] = BC_dirichlet[-1]
        # Here the advection term computation does not include the de-aliasing
        adv = u * u / 2.0
        diff = nu2 * (self.conv(self.D, u)) * self.mesh["Jinv"]
        flux = (adv - diff)

        # Integration
        integral_volume = self.conv(self.Dt, flux * self.weights)
        integral_volume = self.gs_add(integral_volume)
     
        dfluxdx_weak =  - integral_volume
        # Dirichlet BC
        flux_L = flux[0,0]
        flux_R = flux[-1,-1]
        dfluxdx_weak[0,0] += -flux_L
        dfluxdx_weak[-1,-1] += flux_R

        return dfluxdx_weak

