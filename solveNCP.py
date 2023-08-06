import numpy as np
import matplotlib.pyplot as plt

class Solve():
    '''\
            Solve a nonlinear complementarity problem by
            semismooth Newton method with inexact Levenberg-
            -Marquardt-type direction search

            Let F:R^n->R^n be continuously differentiable and
            possibly monotone mapping on R^n. The nonlinear
            complementarity problem is to find x in R^n s.t.
                x_i>=0, F_i(x)>=0, x_i*F_i(x)=0 for all i.

            - Args:
                F: user-defined function F:R^n->R^n
                x0: initial point
                JacF (option): Jacobian of F (None by default)
    '''

    def __init__(self, F, x0, JacF=None, tol=1e-6,\
            alpha=0.5, beta=1e-4, sigma=0.9, rho=1e-6, kmax=1000, verbose=False):
        self.F = F
        self.x0 = x0
        self.n = len(x0)
        if JacF==None:
            # use finite diff
            self.JacF = self.finite_diff
        else:
            # test
            V = self.finite_diff(x0)
            J = JacF(x0)
            if np.linalg.norm(V-J) > self.n*1e-9:
                raise ValueError(f"JacF is not equal to the fd of F: {np.linalg.norm(V-J)}")
            self.JacF = JacF
        self.tol = tol
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.rho = 1e-6
        self.kmax = kmax
        self.verbose = verbose

        # solve nonlinear complementarity problem
        self.x, self.residual, self.gradnorm, self.iter_num = self.solve()

    def finite_diff(self, x):
        eps = 1e-4
        Jacval = np.zeros(self.n*self.n).reshape(self.n,self.n)
        for i in range(self.n):
            for j in range(self.n):
                tmp = x[j]
                x[j] = tmp + eps
                Fp = self.F(x)
                x[j] = tmp - eps
                Fm = self.F(x)
                x[j] = tmp
                Jacval[i,j] = (Fp[i]-Fm[i]) / (2*eps)
        return Jacval

    def compl_set_id(self, x, F):
        set_beta = set()
        eps = 1e-6
        for i in range(self.n):
            if np.abs(x[i]) < eps and np.abs(F[i]) < eps:
                set_beta.add(i)
        return set_beta

    def FBfunc(self, a, b):
        return np.sqrt(a**2+b**2)-a-b

    def BsubFBfunc(self, x, F, JacF): # Bouligand subdiff of FBfunc
        HA = np.zeros((self.n, self.n))
        HB = np.zeros((self.n, self.n))
        set_beta = self.compl_set_id(x, F)
        z_d = np.ones(self.n)
        for i in range(self.n):
            if i not in set_beta:
                HA[i,i] = x[i]/np.sqrt(x[i]**2+F[i]**2) - 1
                HB[i,:] = (F[i]/np.sqrt(x[i]**2+F[i]**2) - 1) * JacF[i,:]
            else:
                HA[i,i] = x[i]/np.sqrt(x[i]**2+(JacF[i,:]@z_d)**2) - 1
                HB[i,:] = (JacF[i,:]@z_d/np.sqrt(x[i]**2+(JacF[i,:]@z_d)**2) - 1) * JacF[i,:]
        return HA + HB

    def meritfunc(self, x, F):
        f = self.FBfunc(x,F)
        return 0.5*np.dot(f,f)

    def gradmerit(self, x, F, BsubFBval):
        return BsubFBval.T@self.FBfunc(x,F)

    def checksearchdirection(self, x, d):
        fdirec = lambda alp: self.meritfunc(x+alp*d,self.F(x+alp*d))
        xrange = [ 0.001 * i for i in range(501) ]
        yrange = [ fdirec(i) for i in xrange ]
        plt.plot(xrange,yrange)
        plt.show()

    def solve(self):
        num_iter = 0
        gamma = 1
        x = self.x0

        Fval = self.F(x)
        FBval = self.FBfunc(x,Fval)
        JacFval = self.JacF(x)
        H = self.BsubFBfunc(x,Fval,JacFval)
        residual = self.meritfunc(x,Fval)
        gradval = self.gradmerit(x,Fval,H)

        while num_iter <= self.kmax and np.linalg.norm(gradval) >= self.tol:
            # For large-scaled NCP, use the following procedure instead
            #if num_iter > 0:
            #    if np.linalg.norm(gradval)/np.linalg.norm(d) > 1 and\
            #            np.linalg.norm(np.minimum(x,Fval)) > 0.1*num_iter*np.sqrt(self.n):
            #        gamma = 0.9**num_iter
            #    else:
            #        gamma = 0
            ## inexact Levenberg--Marquardt-like direction
            #d = np.linalg.solve(H.T@H+gamma*np.eye(self.n),-H.T@FBval+FBval)

            try:
                d = -np.linalg.solve(H,FBval)
            except:
                d = -gradval

            if self.meritfunc(x+d,self.F(x+d)) <= self.sigma*residual:
                x += d
            else:
                # sufficient descent direction?
                if gradval@d > -self.rho*d@d:
                    d = - gradval
                    #print('\033[32m'+"use -grad"+'\033[0m')
                # determine stepsize by Armijo line search
                m = 0
                #self.checksearchdirection(x,d)
                while True:
                    stepsize = self.alpha**m
                    if self.meritfunc(x+stepsize*d, self.F(x+stepsize*d)) <=\
                            residual + self.beta*stepsize*gradval@d:
                        break
                    else:
                        m += 1
                #print(f"m={m}")
                x += stepsize*d

            Fval = self.F(x)
            FBval = self.FBfunc(x,Fval)
            JacFval = self.JacF(x)
            H = self.BsubFBfunc(x,Fval,JacFval)
            residual = self.meritfunc(x,Fval)
            gradval = self.gradmerit(x,Fval,H)

            if self.verbose==True:
                print(f"iter: {num_iter}, grad: {np.linalg.norm(gradval)}, residual: {residual}")

            num_iter += 1

        residual = self.meritfunc(x,self.F(x))
        gradnorm = np.linalg.norm(self.gradmerit(x,self.F(x),self.JacF(x)))
        return x, residual, gradnorm, num_iter 
