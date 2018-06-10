# 12 Steps to Navier-Stokes Equations

## **Challenge**

Use computing capabilities of Python to solve the nonlinear coupled partial derivative equations that govern the dynamics of fluids, the Navier-Stokes equations.

## **Actions**

* creating implicit numerical schemes to solve ever increasing difficult components of the NS equations
* linear convection:

![](.gitbook/assets/2d_linear_conv_initial_conditions.png)

![](.gitbook/assets/2d_linear_conv_solution.png)

* nonlinear convection:

![](.gitbook/assets/nonlinear_2d_solution_1.png)

* diffusion:

![](.gitbook/assets/diffusion_initial_conditions.png)

![](.gitbook/assets/sol_diffusion_10.png)

![](.gitbook/assets/sol_diffusion_30.png)

![](.gitbook/assets/sol_diffusion_270.png)

* Burgers' equation

![](.gitbook/assets/burgers_ic.png)

![](.gitbook/assets/burgers_sol_120.png)

![](.gitbook/assets/burgers_sol_1200.png)

* cavity flow

![](.gitbook/assets/cav_sol_10.png)

![](.gitbook/assets/solution_5000.png)

* channel flow

![](.gitbook/assets/solution.png)

## **Results**

The result of this exercise was package of numerical solutions to the difficult equations of fluid dynamics; the implementation is only in 2D and can solve any problem that can be formulated in a structured 2D mesh; the main equations take also into account turbulence and as seen in the results of teh cavity problem, turbulence is modelled implicitly in the solutions of this project.

For a complete overview of this project please visit its dedicated repository on github:     [https://github.com/FlorinGh/12-steps-to-navier-stokes](https://github.com/FlorinGh/12-steps-to-navier-stokes)​.

