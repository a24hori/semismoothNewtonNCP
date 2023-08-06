# semismoothNewtonNCP

For a differentiable vector-valued function $F\colon\Bbb^n\to\Bbb^n$, the nonlinear complementarity problem (NCP) is defined as

$$
x_i\ge0,\ F_i(x)\ge0,\ x_iF_i(x)=0,\quad i=1,\dots,n.
$$

We implemented a semismooth Newton method for the NCP which is proposed by Luca, Facchinei, and Kanzow (1997) ``A theoretical and numerical comparison of some semismooth algorithms for complementarity problems'', Math. Program.
