- Test that the nonwetting pressure - wetting saturation formulation of the [[Two-phase flow]] model is implemented correctly in #porepy .
- Equations are manually calculated.

#### Setup at $t_0=0$
- Four grid cells
- Homogeneous initial conditions
- Homogeneous Neumann bc ($\nabla p\cdot\mathbf{n}=0$) in the left, top and right
- Homogeneous Dirichlet bc ($p=0$) in the bottom
	- **Note:** The boundary values are enforced **on** the boundary and not on the centers of imaginary grid cells outside the boundary. -> This results, e.g., in the flux being twice of what one would expect otherwise (as the pressure differential is larger).
- For simplicity, all model parameters are set to the following values
	- $\eta_{w}=\eta_{n}=1$
	- $\rho_{w}=\rho_{n}=1$
	- $\phi=1$
	- $\mathbf{k}=I_{2}$
	- $n_{1}=2,n_{2}=3,n_{3}=1\text{ Brooks-Corey rel. perm. model}$
	- $c=0.1\text{ linear cap. pressure model}$
	- $S_{w,res}=0.3,S_{n,res}=0.3$
	- $\hat{S}_{w}=\frac{S_{w}-S_{w,res}}{1-S_{w,res}-S_{n,res}}$
	- $\Delta t=0.2$
- The rel. perm. of the model is not limited below or above. 

#### Manual calculations
The calculations for Jacobian and residual for
- `test_cap_pressure_function`
- `test_total_mobility`
- `test_w_mobility`
- `test_n_mobility`
are rather simple and partially covered by the code.

Note that we can ignore any upwind directions, as the variables are equal at $t=0.$

##### Flux
The flux is calculated with [[TPFA]] (**note,** that in the model we use [[MPFA]], but for easier calculation by hand we switch for testing). Exemplarily, we calculate the residual and Jacobian of an inner interface and a Dirichlet boundary interface by hand. 
**Note,** that we ignore any mobility for `test_flux_n`. Furthermore, permeability and viscosities are set to $1.$
For `I_8` it holds$$\textbf{u}|_{I_8}=\frac{p_{n}(C_{0})-p_{n}(C_{2})}{|C_{0}-C_{2}|}.$$
For `I_6` it holds$$\textbf{u}|_{I_{6}}=\frac{p_{n}(C_{0})-0}{|C_{0}-I_{6}|}.$$
**Note,** that in both equations the **outflow** is calculated, i.e., $\mathbf{u}=-\nabla p_{n}.$

Thus, we obtain a derivative $\left(\frac{\partial\textbf{u}|_{I_{j}}}{\partial p_{j}}\right)_{i,j}$$$\begin{pmatrix}
0 & 0 & 0 & 0 \\
1 & -1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 1 & -1 \\
0 & 0 & 0 & 0 \\
2 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
-1 & 0 & 1 & 0 \\
0 & -1 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{pmatrix}\tag{1}.$$Summing over the interfaces bordering each volume (and multiplying with the correct sign, i.e., the direction of the face normal), we obtain the Jacobian of $\left(\frac{\partial\nabla\cdot\textbf{u}|_{C_{j}}}{\partial p^{j}}\right)_{i,j}$ $$\begin{pmatrix}
4 & -1 & -1 & 0 \\
-1 & 4 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}.$$

##### Flow equation
The flow equation (without gravity) is given by $$\nabla\cdot\left(-\lambda_t\nabla p_n+\lambda_w\nabla p_c\right)=q_t.$$Discretized in time this still reads the same. We look at the Jacobian for $S_{w}$ and $p_{n}$ individually, while fixing the other one.

As it holds $S_{w}^{i}|_{t=0}=0.5$ for all volumes $V_{i},$ we have $p_{c}(S_{w})=c\hat{S}_{w}=0.1\cdot \hat{S_{w}}=0.1\cdot\left(\frac{S_{w} - 0.3}{0.4}\right).$ Thus, if we ignore $\lambda_{n},$ $\left(\frac{\partial\nabla\cdot\textbf{u}_{p_{c}}|_{C_{j}}}{\partial S_{w}^{j}}\right)_{i,j}$ is given by $$\frac{0.1}{0.4}\begin{pmatrix}
2 & -1 & -1 & 0 \\
-1 & 2 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}.$$**Note,** that **no** capillary flux takes place across boundaries.

If we take $\lambda_{n}$ into account, it gets more difficult. In the Brooks-Corey model, it holds $\lambda_{w}(S_{w})=\frac{k_{r,w}(\hat{S}_w)}{\eta_{w}},$ i.e., $$
\begin{align}
	\lambda_{w}(S_{w}) & = \left(\frac{S_{w} - S_{w,rel}}{1 - S_{w,rel} - S_{n,rel}}\right)^{5} \\
	& = \left(\frac{S_{w} - 0.3}{0.4}\right)^{5} \\
	& \implies \lambda_{n}(0.5) = (0.5)^{5} = 0.03125
\end{align}	
$$We compute$$\begin{align}
	\frac{\partial\left[\nabla\cdot(-\lambda_{t}\nabla p_{n} + \lambda_{w}\nabla p_{c}) - q_{t}\right]^{i}}{\partial S_{w}^{j}} & =
	\frac{\partial\left[\nabla\cdot(\lambda_{w}\nabla p_{c})\right]^{i}}{\partial S_{w}^{j}} \\
	& = \underbrace{\sum\limits_{k}\nabla p_{c}^{k} \partial_{S_{w}^{j}}\lambda_{w}^{k}}_{=0} + \sum\limits_{k}\lambda_{w}^{k}\partial_{S_{w}^{j}}\nabla p_{c}^{k} \\
	& = \sum\limits_{k}\lambda_{w}^{k}\partial_{S_{w}^{j}}\nabla p_{c}^{k},
\end{align}$$where we number the interfaces adjacent to $V_{i}$ with $k.$ This holds since $p_{n}^{i}$ and $p_{c}^{i}$ are equal for all volumes, hence the flux terms vanishes when using TPFA. Also, there is no out- or inflow at the Dirichlet boundary at $t=0.$

Putting all calculations together, we deduct that the Jacobian equals $$0.03125 \cdot\frac{0.1}{0.4}\cdot(-1)
\begin{pmatrix}
2 & -1 & -1 & 0 \\
-1 & 2 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}.$$

As $p_{c}$ is independent of $p_{n},$ we can ignore it when calculating the derivative of the flow equation w.r.t. $p_{n}.$ We compute$$\begin{align}
\frac{\partial\left(\nabla\cdot( - \lambda_t\nabla p_{n} + \lambda_n\nabla p_c)-q_t\right)}{\partial p_{n}^{j}}^{i} & = \frac{\partial\left(\nabla\cdot( - \lambda_t\nabla p_{n})\right)^{i}}{\partial p_{n}^{j}} \\
\end{align}.$$Note that $\lambda_{t}^{i}=0.5$ for the southern outer boundaries and $\hat{S}_{w}^5+(1-\hat{S}_{w})^2(1-\hat{S}_{w}^{3})=0.5^{5}+0.25\cdot0.875=0.25.$ Hence we find that the Jacobian is given by $$0.25\cdot\begin{pmatrix}
4 & -1 & -1 & 0 \\
-1 & 4 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}+0.5\cdot\begin{pmatrix}
2 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{pmatrix}.$$

The residual of the flow equation computes as $$\left(\nabla\cdot\left(-\lambda_t\nabla p_n+\lambda_n\nabla p_c\right)-q_{t}\right)^{i}=-q_{t}^{i}
$$This holds since $p_{n}^{i}$ and $p_{c}^{i}$ are equal for all volumes, hence the terms vanish when using TPFA. **Note,** that $p_{c}$ has homogeneous Neumann bc. $p_{n}$ has homogeneous Dirichlet bc from the southern boundary, but since the internal pressure equals $0,$ this results in no flow.

##### Transport equation
The transport equation reads$$\phi\frac{\partial S_w}{\partial t}+\nabla\cdot\left(f_w\mathbf{u}+f_w\lambda_n\nabla(p_c+\Delta\rho\mathbf{g})\right)=q_w,$$and must first be discretized in time by implicit Euler$$\phi\frac{S_{w}^{n+1}-S_{w}^{n}}{\Delta t}+\nabla\cdot\left(f_w(S_{w}^{n+1})\mathbf{u}+f_w(S_{w}^{n+1})\lambda_n\nabla(p_c(S_{w}^{n+1}))\right)-q_w,$$where
$$\mathbf{u}=-\lambda_t(S_{w}^{n+1})\nabla p_n+\lambda_w(S_{w}^{n+1})\nabla p_c.
$$
The 

As $p_{c}$ and $S_{w}$ are independent of $p_{n},$ we can ignore the corresponding terms when calculating the derivative of the flow equation w.r.t. $p_{n}.$ We compute$$
\begin{align}
\frac{\partial\left[\phi\frac{S_{w}^{n+1} - S_{w}^{n}}{\Delta t}+\nabla\cdot(f_{w}(-\lambda_{t}\nabla p_{n} + \lambda_{w}\nabla p_{c}) + f_{w}\lambda_n\nabla p_{c}) - q_{t}\right]}{\partial p_{n}^{j}}^{i} & = \frac{\partial\left(\nabla\cdot( - \lambda_{w}\nabla p_{n})\right)^{i}}{\partial p_{n}^{j}} \\
\end{align}.$$Note that $\lambda_{w}^{i} = \frac{0.5}{0.5+0.5} = 0.25$ for the southern outer boundaries and $\lambda_{w}^{i} = \hat{S}_{w}^{5} = 0.5^{5} = 0.03125$ for the inner interfaces. Hence we find that the Jacobian is given by $$0.03125\cdot
\begin{pmatrix}
2 & -1 & -1 & 0 \\
-1 & 2 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}
+ 0.25
\cdot\begin{pmatrix}
2 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{pmatrix}.$$
Now, we compute the derivative w.r.t. $S_{w}^{n+1}$ $$
\begin{align}
	& \frac{\partial\left[\phi\frac{S_{w}^{n+1} - S_{w}^{n}}{\Delta t}+\nabla\cdot(f_{w}(-\lambda_{t}\nabla p_{n} + \lambda_{w}\nabla p_{c}) + f_{w}\lambda_n\nabla p_{c}) - q_{t}\right]}{\partial S_{w}^{n+1,j}}^{i} \\
	& = \frac{\phi}{\Delta t}\delta_{i,j} + \frac{\partial\left(\nabla\cdot(f_{w}(\lambda_{w} + \lambda_{n})\nabla p_{c})\right)^{i}}{\partial S_{w}^{j,n+1}} \\
	& = \frac{\phi}{\Delta t}\delta_{i,j} + \sum\limits_{k}\frac{\partial\left(\lambda_{w}\nabla p_{c})\right)^{k}}{\partial S_{w}^{j,n+1}} \\
	& = \frac{\phi}{\Delta t}\delta_{i,j} + \sum\limits_{k}\nabla p_{c}^{k}\frac{\partial\lambda_{w}^{k}}{\partial S_{w}^{j,n+1}} + \lambda_{w}^{k}\frac{\partial(\nabla p_{c})^{k}}{\partial S_{w}^{j,n+1}}\\
	& = \frac{\phi}{\Delta t}\delta_{i,j} + \sum\limits_{k}\lambda_{w}^{k}\frac{\partial(\nabla p_{c})^{k}}{\partial S_{w}^{j,n+1}}
\end{align}$$where we number the interfaces adjacent to $V_{i}$ with $k.$ As the capillary pressure induces no flux at the domain boundaries (homogeneous Neumann bc), the Jacobian is equal to$$\frac{\phi}{\Delta t}I_{4} + 0.03125\cdot\frac{0.1}{0.4}\cdot(-1)
\begin{pmatrix}
2 & -1 & -1 & 0 \\
-1 & 2 & 0 & -1 \\
-1 & 0 & 2 & -1 \\
0 & -1 & -1 & 2 \\
\end{pmatrix}.$$

The residual of the transport equation computes as $$
\begin{align}
\left[\phi\frac{S_{w}^{n+1}-S_{w}^{n}}{\Delta t}+\nabla\cdot\left(f_w(S_{w}^{n+1})\mathbf{u}+f_w(S_{w}^{n+1})\lambda_n\nabla(p_c(S_{w}^{n+1}))\right)-q_w\right]^{i} = -q_{w}^{i}
\end{align}$$since the initial guess for $S_{w}^{n+1}$ equals $S_{w}^{n}$ and all fluxes are zero.
