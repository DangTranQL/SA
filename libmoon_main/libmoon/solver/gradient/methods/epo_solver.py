# import numpy as np
# import cvxpy as cp
# import cvxopt
# import torch
# import warnings
# from matplotlib import pyplot as plt

# from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
# from libmoon.problem.synthetic.zdt import ZDT1

# warnings.filterwarnings("ignore")


# # -------------------------- Helpers --------------------------

# def _finite_real(x):
#     """Cast to float64 and replace NaN / +/-inf with large finite values."""
#     x = np.asarray(x, dtype=np.float64)
#     return np.nan_to_num(x, nan=0.0, posinf=1e12, neginf=-1e12)


# def mu(rl, normed=False):
#     # robust entropy-like measure
#     rl = np.clip(rl, 0, np.inf)
#     m = len(rl)
#     l_hat = rl if normed else rl / rl.sum()
#     eps = np.finfo(rl.dtype).eps
#     l_hat = l_hat[l_hat > eps]
#     return np.sum(l_hat * np.log(l_hat * m))


# def adjustments(l, r=1):
#     # (rl, mu_rl, a) as in the paper/code
#     m = len(l)
#     rl = r * l
#     l_hat = rl / rl.sum()
#     mu_rl = mu(l_hat, normed=True)
#     eps = 1e-3  # avoid log(0)
#     a = r * (np.log(np.clip(l_hat * m, eps, np.inf)) - mu_rl)
#     return rl, mu_rl, a


# # -------------------------- EPO LP core --------------------------

# class EPO_LP(object):
#     """
#     Paper:
#       - https://proceedings.mlr.press/v119/mahapatra20a.html
#       - https://arxiv.org/abs/2010.06313
#     """

#     def __init__(self, m, n, r, eps=1e-4, solver='SCS', debug=False):
#         cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
#         self.m, self.n, self.r, self.eps = m, n, r, eps
#         self.solver = solver        # 'SCS' (default), 'ECOS', or 'GLPK'
#         self.debug = debug
#         self.last_move = None

#         # LP parameters / variables
#         self.a = cp.Parameter(m)            # adjustments
#         self.C = cp.Parameter((m, m))       # Gram matrix (G^T G)
#         self.Ca = cp.Parameter(m)           # C @ a
#         self.rhs = cp.Parameter(m)          # RHS for balancing
#         self.alpha = cp.Variable(m)         # simplex weights

#         # Balance LP: maximize alpha^T Ca subject to simplex and C alpha >= rhs
#         obj_bal = cp.Maximize(self.alpha @ self.Ca)
#         constr_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1, self.C @ self.alpha >= self.rhs]
#         self.prob_bal = cp.Problem(obj_bal, constr_bal)

#         # Dominance LPs
#         obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))
#         constr_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,
#                       self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
#                       self.C @ self.alpha >= 0]
#         constr_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1, self.C @ self.alpha >= 0]
#         self.prob_dom = cp.Problem(obj_dom, constr_res)
#         self.prob_rel = cp.Problem(obj_dom, constr_rel)

#         self.gamma = 0.0   # last optimum value
#         self.mu_rl = 0.0   # last non-uniformity

#     def _solve_prob(self, prob):
#         """Try preferred solver, then fallbacks."""
#         order = [self.solver] + [s for s in ['SCS', 'ECOS', 'GLPK'] if s != self.solver]
#         for s in order:
#             try:
#                 if s == 'GLPK':
#                     return prob.solve(solver=cp.GLPK, verbose=self.debug, warm_start=True)
#                 if s == 'ECOS':
#                     return prob.solve(solver=cp.ECOS, verbose=self.debug, warm_start=True)
#                 # SCS default (more forgiving for LPs with tight numerics)
#                 return prob.solve(solver=cp.SCS, verbose=self.debug, warm_start=True,
#                                   max_iters=20000, eps=1e-6)
#             except cp.SolverError:
#                 if self.debug:
#                     print(f"[EPO_LP] {s} failed; trying next solver...")
#                 continue
#         return None  # all solvers failed

#     def get_alpha(self, l, G, r=None, C=False, relax=False):
#         """
#         l: (m,) losses
#         G: (m,m) if C=True (already Gram), else (m,n) Jacobian, will be G G^T
#         r: (m,) reweight vector
#         """
#         r = self.r if r is None else r

#         # sanitize
#         l = _finite_real(l)
#         r = _finite_real(r)
#         G = _finite_real(G)

#         assert len(l) == len(r) == self.m, "length(l), length(r) must equal m"

#         # build Gram matrix
#         if C:
#             assert G.shape == (self.m, self.m), "When C=True, G must be (m,m)"
#             Cmat = G
#         else:
#             assert G.shape[0] == self.m, "When C=False, G must be (m,n)"
#             Cmat = G @ G.T

#         Cmat = _finite_real(Cmat) + 1e-12 * np.eye(self.m)  # tiny ridge

#         # adjustments
#         rl, self.mu_rl, a = adjustments(l, r)
#         a = _finite_real(a)
#         Ca = _finite_real(Cmat @ a)

#         # assign parameters
#         self.C.value = Cmat
#         self.a.value = a
#         self.Ca.value = Ca

#         if self.debug:
#             print("nan/inf in C:", np.isnan(Cmat).any(), np.isinf(Cmat).any())
#             print("nan/inf in a:", np.isnan(a).any(), np.isinf(a).any())
#             print("nan/inf in Ca:", np.isnan(Ca).any(), np.isinf(Ca).any())

#         NEG_BIG = -1e12

#         if self.mu_rl > self.eps:
#             # balance step
#             J = Ca > 0
#             if np.any(J):
#                 J_star_idx = np.where(rl == np.max(rl))[0]
#                 rhs = Ca.copy()
#                 rhs[J] = NEG_BIG           # instead of -np.inf
#                 rhs[J_star_idx] = 0.0
#             else:
#                 rhs = np.zeros_like(Ca)
#             self.rhs.value = _finite_real(rhs)

#             self.gamma = self._solve_prob(self.prob_bal)
#             self.last_move = "bal"
#         else:
#             # dominance (or relaxed)
#             prob = self.prob_rel if relax else self.prob_dom
#             self.gamma = self._solve_prob(prob)
#             self.last_move = "dom"

#         if self.gamma is None:
#             if self.debug:
#                 print("[EPO_LP] All solvers failed; returning None alpha.")
#             return None
#         return self.alpha.value


# # -------------------------- Glue to LibMOON solver --------------------------

# def solve_epo(grad_arr, losses, pref, epo_lp):
#     """
#     grad_arr: (m, n) Jacobian (PyTorch tensor)
#     losses:   (m,) losses (tensor/ndarray)
#     pref:     (m,) preference weights (tensor/ndarray)
#     epo_lp:   EPO_LP instance
#     returns:  gw (tensor, shape n), alpha (ndarray, shape m)
#     """
#     if isinstance(pref, torch.Tensor):
#         pref = pref.detach().cpu().numpy()
#     pref = _finite_real(pref)

#     G = grad_arr.detach().cpu().numpy()
#     G = _finite_real(G)

#     if isinstance(losses, torch.Tensor):
#         losses_np = losses.detach().cpu().numpy().squeeze()
#     else:
#         losses_np = losses
#     losses_np = _finite_real(losses_np)

#     GG = G @ G.T  # (m x m) Gram
#     alpha = epo_lp.get_alpha(losses_np, G=GG, C=True)
#     if alpha is None:  # fallback if LP fails
#         alpha = pref / pref.sum()
#     gw = alpha @ G
#     return torch.tensor(gw, dtype=torch.float32), alpha


# class EPOCore:
#     def __init__(self, n_var, prefs, solver='SCS', debug=False):
#         """
#         prefs: (n_prob, n_obj)
#         """
#         self.core_name = 'EPOCore'
#         self.prefs = prefs
#         self.n_prob, self.n_obj = (prefs.shape[0], prefs.shape[1])
#         self.n_var = n_var

#         prefs_np = prefs.cpu().numpy() if isinstance(prefs, torch.Tensor) else prefs
#         # Build one LP per preference vector
#         self.epo_lp_arr = [
#             EPO_LP(m=self.n_obj, n=self.n_var, r=1.0 / pref, solver=solver, debug=debug)
#             for pref in prefs_np
#         ]

#     def get_alpha(self, Jacobian, losses, idx):
#         _, alpha = solve_epo(Jacobian, losses, self.prefs[idx], self.epo_lp_arr[idx])
#         return torch.tensor(alpha, dtype=torch.float32)


# class EPOSolver(GradBaseSolver):
#     def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
#                  folder_name=None, verbose=True, lp_solver='SCS', lp_debug=False):
#         self.folder_name = folder_name
#         self.solver_name = 'EPO'
#         self.problem = problem
#         self.prefs = prefs
#         self.epo_core = EPOCore(n_var=problem.n_var, prefs=prefs,
#                                 solver=lp_solver, debug=lp_debug)
#         # GradBaseSolver(step_size, epoch, tol, core, verbose=True)
#         super().__init__(step_size, n_epoch, tol, self.epo_core, verbose=verbose)

#     def solve(self, x_init):
#         # delegates to GradBaseSolver
#         return super().solve(self.problem, x_init, self.prefs)


# # -------------------------- Demo / __main__ --------------------------

# if __name__ == '__main__':
#     # Problem setup
#     n_obj, n_var, n_prob = 2, 10, 8
#     problem = ZDT1(n_var=n_var)

#     # Create a simple set of preferences on the 2D simplex
#     pref_1d = np.linspace(0.1, 0.9, n_prob)
#     prefs = np.c_[pref_1d, 1 - pref_1d].astype(np.float64)

#     # Build solver (SCS is default; set lp_solver='GLPK' if you have it)
#     solver = EPOSolver(problem=problem,
#                        prefs=prefs,
#                        step_size=1e-2,
#                        n_epoch=1000,
#                        tol=1e-2,
#                        verbose=True,
#                        lp_solver='SCS',   # 'SCS' | 'ECOS' | 'GLPK'
#                        lp_debug=False)    # set True to print LP diagnostics

#     # Initial points: shape (n_prob, n_var)
#     x0 = torch.rand(n_prob, n_var)
#     res = solver.solve(x_init=x0)

#     # Plot results in objective space
#     y_arr = res['y']
#     plt.figure(figsize=(5, 4))
#     plt.scatter(y_arr[:, 0], y_arr[:, 1], s=12)
#     plt.xlabel('f1')
#     plt.ylabel('f2')
#     plt.title('EPO (ZDT1)')
#     plt.tight_layout()
#     plt.show()



import numpy as np
import cvxpy as cp
import cvxopt
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
import torch
import warnings
warnings.filterwarnings("ignore")
from libmoon.problem.synthetic.zdt import ZDT1
from matplotlib import pyplot as plt


class EPO_LP(object):
    # Paper:
    # https://proceedings.mlr.press/v119/mahapatra20a.html,
    # https://arxiv.org/abs/2010.06313
    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing
        self.alpha = cp.Variable(m)  # Variable to optimize
        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance
        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance
        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"
        return self.alpha.value


def mu(rl, normed=False):
    # Modified by Xiaoyuan to handle negative issue.
    # if len(np.where(rl < 0)[0]):
    #     raise ValueError(f"rl<0 \n rl={rl}")
    #     return None
    rl = np.clip(rl, 0, np.inf)
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))

def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    eps = 1e-3  # clipping by eps is to avoid log(0), zxy Dec. 5.
    a = r * (np.log(np.clip(l_hat * m, eps, np.inf)) - mu_rl)
    return rl, mu_rl, a

def solve_epo(grad_arr, losses, pref, epo_lp):
    '''
        input: grad_arr: (m,n).
        losses : (m,).
        pref: (m,) inv.
        return : gw: (n,). alpha: (m,)
    '''
    if type(pref) == torch.Tensor:
        pref = pref.cpu().numpy()
    pref = np.array(pref)
    G = grad_arr.detach().clone().cpu().numpy()
    if type(losses) == torch.Tensor:
        losses_np = losses.detach().clone().cpu().numpy().squeeze()
    else:
        losses_np = losses
    m = G.shape[0]
    n = G.shape[1]
    GG = G @ G.T
    alpha = epo_lp.get_alpha(losses_np, G=GG, C=True)
    if alpha is None:  # A patch for the issue in cvxpy
        alpha = pref / pref.sum()
    gw = alpha @ G
    return torch.Tensor(gw), alpha

class EPOSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3, folder_name=None,verbose = True):
        self.folder_name = folder_name
        self.verbose = verbose
        self.solver_name = 'EPO'
        self.problem = problem
        self.prefs = prefs
        self.epo_core = EPOCore(n_var=problem.n_var, prefs=prefs)
        super().__init__(step_size, n_epoch, tol, self.epo_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)


class EPOCore():
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'EPOCore'
        self.prefs = prefs
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        self.n_var = n_var
        prefs_np = prefs.cpu().numpy() if type(prefs) == torch.Tensor else prefs
        self.epo_lp_arr = [EPO_LP(m=self.n_obj, n = self.n_var, r=1/pref) for pref in prefs_np]

    def get_alpha(self, Jacobian, losses, idx):
        _, alpha = solve_epo(Jacobian, losses, self.prefs[idx], self.epo_lp_arr[idx])
        return torch.Tensor(alpha)


if __name__ == '__main__':
    n_obj, n_var, n_prob = 2, 10, 8
    # prefs = np.random.rand(n_prob, n_obj)
    pref_1d = np.linspace(0.1, 0.9, n_prob)
    prefs = np.c_[pref_1d, 1 - pref_1d]
    problem = ZDT1(n_var=n_var)
    solver = EPOSolver(step_size=1e-2, n_epoch=1000, tol=1e-3, problem=problem, prefs=prefs)
    x0 = torch.rand(n_prob, n_var)
    res = solver.solve(x_init=x0)
    y_arr = res['y']
    plt.scatter(y_arr[:, 0], y_arr[:, 1], color='black')
    plt.title('Results of EPOSolver')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.grid()
    plt.show()
