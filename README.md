# 3DIncompressibleSG

This code was designed to implement a solver for the incompressible semi-geostrophic equations in three dimensions using the geometric method. 

## Necessary packages

This code uses the standard suite of python packages, namely, matplotlib, numpy, and scipy. In addition to this this code uses the msgpack data format to store the output efficiently. 

### Specialty packages:

The core specialty package of this code is the sd-ot/pysdot package created by Merigot and Leclerc to solve the optimal transport problem.

The other specialty package is PETSc which is used as the linear solver in the optimal transport problem. To install PETSc on ubuntu with root access do the following:
1. From your terminal in your home folder begin with:
    ~~~
    mkdir -p ~/Software
    ~~~
2. And enter the folder: 
    ~~~
    cd Software
    ~~~
3. Clone the github repository for PETSc
    ~~~
    git clone -b release https://github.com/petsc/petsc.git petsc
    ~~~
4. Next:
    ~~~
    cd petsc
    ~~~
5. Next:
    ~~~
    ./configure --with-petsc4py=1
    ~~~
6. Next:
    ~~~bash
    make PETSC_DIR=/home/ . . . /Software/petsc PETSC_ARCH=linux-c-opt all
    ~~~
7. Ensure that the installation was successful with:
    ~~~
    make PETSC_DIR=/home/ . . . /Software/petsc PETSC_ARCH=linux-c-opt check
    ~~~
8. Modify .bashrc or .zshrc with the path to your PETSc installation:
    ~~~
    # >>> PETSc initialize >>>
    export PETSC_DIR="/home/ . . . /Software/petsc"
    export PETSC_ARCH="linux-c-opt"
    export PYTHONPATH="/home/ . . . /Software/petsc/linux-c-opt/lib:$PYTHONPATH"
    # <<< PETSc initialize <<<
    ~~~

Make sure that you have updated the key packages such as pip, numpy, scipy, and matplotlib as well as your compiler such as gcc or g++.

## Usage

A demonstration of how the code is used is presented in _RunCode.ipynb_. 

### Key Features:

_weightguess.py_ is the implementation of a rescalling scheme to ensure that the initial guess for the Damped Newton Method used to solve the optimal transport problem is optimal. This is done by translating and rescalling the seeds (read diracs of the target measure) so that they are inside the source domain (in this code written throughout as 'box' or 'bx') and then generating the weights that give a Voronoi tesselation. These weights are then fed into the optimal transport solver. 

_optimaltransportsolver.py_ is the implementation of the optimal transport solver. It is broken into two parts. The first part is the construction of the optimal transport domain, taking into account the possibility for anisotropic periodicity. The second part is calling the optimal transport solver from the pysdot package and creating replications of the seed positions to account for the type of periodicity required. 

_main.py_ is the implementation of Adams-Bashforth 2 to solve the ODE and call all the requisit pieces. 

## Initial Conditions

At the moment there are two initial conditions implemented. They can be found in _initialconditions.py_.

### Evolution of an Isolated Semi-Geostrophic Cyclone

This initial condition is an implementation of the work of Schaer and Wernli. In order to construct the initial condition we use FEniCS to solve Laplace's equation
$$\Delta \tilde{\Phi}=0$$
for the perturbation $ \tilde{\Phi}(x,y,z)$. The perturbation is given in a box $[-a,a]\times[-b,b]\times[0,c]$ which periodic in $x$ and $y$ and has Neumann boundary conditions in $z$. The Neumann boundary conditions are given as:
$$\frac{\partial\tilde{\Phi}}{\partial z}\bigg\vert_{z=0}=f(x,y)\quad\mathrm{and}\quad \frac{\partial\tilde{\Phi}}{\partial z}\bigg\vert_{z=c}=g(x,y).$$
The functions $f(x,y)$ and $g(x,y)$ are structurally similar in that they are both composed of three circular perturbations but differ in the offset and amplitude of the circular perturbations.

An explicit solution to this partial differential equation is possible to construct using a Fourier expansion. However, the Fourier coefficients cannot be found analytically. In _testspace.ipynb_ one can find a numerical implementation finding the first of the needed Fourier coefficients. 

Once the solution to Laplace's equation for the perturbation is obtained we add it to the base state,
$$\overline{\Phi}(x,y,z) = \frac{1}{2} \left(\arctan\left(\frac{y}{1 + z}\right) - \arctan\left(\frac{y}{1 - z}\right)\right) - 0.12yz - \frac{1}{2}A\left(y^2 - z^2\right),$$
where $A$ is the shear parameter. Thus we recover the full modified geopotential function
$$\Phi=\overline\Phi+\tilde\Phi$$
and the intial condition is taken to be the pushforward of
$$\text{Id}+\nabla\Phi.$$

### Perturbation of a Steady State

The other initial condition that is implemented is the perturbation of a basic state. This follows from the work of Charles Egan. A steady state can be constructed out of any $3\times3$ symmetric invertible matrix, $B$. This corresponds to a choice of modified geopotential of the form 
$$\overline P=\frac{1}{2}x\cdot Bx$$
To construct these initial conditions we take the steady state encoded by $B$ and map the fluid (read source) domain forward under
$$\nabla \overline P = Bx$$
to find the geostrophic (read target) domain. The geostrophic domain is then filled with a lattice that is mapped back to the fluid domain via $\left(\nabla \overline P\right)^{-1}$. A perturbation is then added to $\overline P$ to create $P$ and the lattice in the fluid domain is then mapped back to the geostrophic domain by $\nabla P$ applying the perturbation and creating the initial seed configuration. 

## Diagnostic Tools

There are two classes of diagnostic tools avaliable to users of this code. 

### Debug Commands

The _debug_ option of the solver can be set to _True_ to force the code to print out the timestep index and the accuracy of the optimal transport solver at every timestep.

The _verbosity_ option of the optimal transport solver itself can be set to 1 or 2 revealing information about which linear solver is being used and the progress of the optimal transport solver as it iterates throught the Damped Newton method.

### Comparisions

In the file _erroranalysis.py_ are several functions that are used in _DataAnalysis.ipynb_ to check if reducing the size of the timestep and increasing the number of seeds is improving the accuracy of the code with respect to a high resolution run of the code. The two key tools here are the Weighted Euclidean Error and the Square-Root of the Sinkhorn Loss. Note that the Square-Root of the Sinkhorn Loss is implemented using the OTT-JAX python package (https://ott-jax.readthedocs.io/en/latest/). 

The other options for comparison is to compute the RMSV, the energy, and the error in the energy with respect to the mean and compare to the results of Egan et al. and J-D Benamou et al. Computation of these properties is preformed by the _Properties_ function in _auxfunctions.py_ and the _Root_Mean_Squared_Velocity_ function in _erroranalysis.py_.