# semismoothNewtonNCP

For a differentiable vector-valued function $F\colon\mathbb{R}^n\to\mathbb{R}^n$, the nonlinear complementarity problem (NCP) is defined as

$$
x_i\ge0,\ F_i(x)\ge0,\ x_iF_i(x)=0,\quad i=1,\dots,n.
$$

We implemented a semismooth Newton method for the NCP which is proposed by Luca, Facchinei, and Kanzow (1997) ``A theoretical and numerical comparison of some semismooth algorithms for complementarity problems'', Math. Program.

# How to use

Input (necessary):

1. The function $F$ (user defined);
2. The initial point $x_0\in\mathbb{R}^n$.

Option:

- You can also define the Jacobian matrix $JF$ of $F$ by `JacF=JF`, for example.

Output:

1. solution `solveNCP.Solve.x`;
2. residual `SolveNCP.Solve.residual`;
3. $\ell_2$-norm of gradient for the vector-valued residual function `solveNCP.Solve.gradnorm`. (In this code, we use the Fischer--Burmeister function);
4. iteration `SolveNCP.Solve.iter_num`.

For example,

```
import numpy as np
import solveNCP

def F(x):
    ''' define myfunc F:R^n->R^n '''

def JF(x):
    ''' (option) Jacobian of F '''

n = 10
x0 = np.zeros(n)
res = solveNCP.Solve(F,x0,JacF=JF)
```

You can set more detailed parameter in the algorithm but it may be complicated to explain here.
We then just introduce how to use the software only.
If you want to know more, read my code or the paper cited above.
