# Physics-Informed Neural Network for Internal Pipe Flow Simulation

**A Physics-Informed Neural Network (PINN) approach to simulating steady, incompressible flow inside a cylindrical pipe using PyTorch.**

## Overview

This project investigates the application of Physics-Informed Neural Networks (PINNs) to the simulation of internal fluid flow. The model represents the flow field in cylindrical coordinates and predicts the radial, azimuthal, and axial velocity components, together with the pressure distribution.

Unlike conventional CFD approaches that typically rely on spatial discretization and mesh generation, PINNs incorporate physical constraints into neural network training through a loss function. The project explores this approach for pipe-flow modeling and visualizes the predicted flow field, pressure variation, and friction-factor distribution.

**Project period:** February 2025 – July 2025

**Course:** Fluid Mechanics I
**Institution:** Sharif University of Technology

## Objectives

* Formulate an internal pipe-flow problem in cylindrical coordinates.
* Implement a physics-informed neural network using Python and PyTorch.
* Generate collocation points within the computational domain.
* Incorporate continuity and prescribed boundary conditions into the training loss.
* Predict and visualize velocity components and pressure.
* Investigate pressure variation along the pipe centerline.
* Estimate the friction factor from the predicted pressure gradient.

## Physical Model

The computational domain represents a straight cylindrical pipe with a circular cross-section. The flow is modeled under steady, incompressible, and laminar-flow assumptions.

The problem is expressed in cylindrical coordinates:

* \(r\): Radial coordinate
* \(\theta\): Azimuthal coordinate
* \(z\): Axial coordinate

The neural network receives the spatial coordinates as inputs and predicts four physical quantities:

$$
\mathbf{u}(r,\theta,z)=
\begin{bmatrix}
u_r & u_\theta & u_z
\end{bmatrix}
$$

$$
p=p(r,\theta,z)
$$

where \(u_r\), \(u_\theta\), and \(u_z\) denote the radial, azimuthal, and axial velocity components, respectively, and \(p\) denotes pressure.

### Governing Physics

The incompressibility constraint is expressed by the continuity equation in cylindrical coordinates:

$$
\frac{1}{r}\frac{\partial (ru_r)}{\partial r}
+\frac{1}{r}\frac{\partial u_\theta}{\partial\theta}
+\frac{\partial u_z}{\partial z}=0
$$

The intended physical model is based on the Navier–Stokes equations for incompressible Newtonian flow. The neural network is trained using physical constraints and prescribed boundary conditions rather than relying exclusively on labeled simulation data.

## Computational Parameters

| Parameter                       | Symbol   |            Value |
| ------------------------------- | -------- | ---------------: |
| Pipe radius                     | \(R\)    |           0.01 m |
| Pipe length                     | \(L\)    |            1.0 m |
| Dynamic viscosity               | \(\mu\)  | \(10^{-3}\) Pa·s |
| Fluid density                   | \(\rho\) |        1.0 kg/m³ |
| Pipe diameter                   | \(D\)    |           0.02 m |
| Prescribed inlet axial velocity | \(u_0\)  |          1.0 m/s |
| Number of collocation points    | \(N_c\)  |           10,000 |
| Optimizer                       | —        |             Adam |
| Learning rate                   | —        |      \(10^{-3}\) |
| Training epochs                 | —        |           10,000 |

## Neural Network Architecture

The implementation uses a fully connected feedforward neural network built with PyTorch.

| Component           | Configuration                     |
| ------------------- | --------------------------------- |
| Input layer         | 3 neurons: \(r,\theta,z\)         |
| Hidden layers       | 2 layers, 64 neurons each         |
| Activation function | Tanh                              |
| Output layer        | 4 neurons: \(u_r,u_\theta,u_z,p\) |
| Optimizer           | Adam                              |
| Training iterations | 10,000 epochs                     |

The network uses automatic differentiation to calculate derivatives of its predictions with respect to the input coordinates.

### Collocation Points

A set of 10,000 collocation points is generated throughout the cylindrical computational domain using Latin Hypercube Sampling (LHS), implemented with `pyDOE`.

These points provide the spatial locations at which the physics-based constraints are evaluated during training.

## Boundary Conditions

The implementation defines the following boundary conditions:

* **Inlet (\(z=0\)):** The axial velocity is prescribed as \(u_z=1.0\) m/s.
* **Outlet (\(z=L\)):** The pressure is prescribed as \(p=0\), serving as a reference pressure.

The report also discusses the no-slip condition at the pipe wall. Explicit enforcement of the wall condition is a necessary consideration when extending the implementation to a more complete physical PINN model.

## Training Methodology

The model is trained by minimizing a loss function that combines the incompressibility constraint with the prescribed inlet and outlet conditions.

The main steps are:

1. Define the pipe geometry and fluid properties.
2. Generate collocation points in cylindrical coordinates.
3. Construct the neural network and its output variables.
4. Evaluate the continuity residual using automatic differentiation.
5. Calculate the boundary-condition losses.
6. Optimize the network parameters using Adam.
7. Evaluate the trained model over a grid of spatial coordinates.
8. Generate contour plots and post-process the pressure distribution.

The loss is printed every 500 epochs to monitor the training process.

## Results and Visualization

The project generates several visualizations to investigate the predicted flow field.

### 1. Radial Velocity Component

The radial velocity contour illustrates the predicted radial motion of the fluid throughout the pipe domain.

**Suggested figure:** `images/radial_velocity.png`

### 2. Azimuthal Velocity Component

The azimuthal velocity contour represents the predicted circumferential component of the flow.

**Suggested figure:** `images/azimuthal_velocity.png`

### 3. Axial Velocity Component

The axial velocity contour visualizes the dominant flow component along the pipe axis and its variation with radial and axial position.

**Suggested figure:** `images/axial_velocity.png`

### 4. Pressure Distribution

The pressure contour provides a spatial representation of the predicted pressure field.

**Suggested figure:** `images/pressure_contour.png`

### 5. Pressure Along the Pipe Centerline

The pressure distribution is evaluated along the pipe centerline and plotted as a function of the axial coordinate.

**Suggested figure:** `images/centerline_pressure.png`

### 6. Friction Factor

The pressure gradient is estimated numerically, and a friction-factor distribution is calculated using the pipe diameter, fluid density, and predicted velocity scale.

**Suggested figure:** `images/friction_factor.png`

## Technologies Used

* **Python** — Numerical computing and implementation
* **PyTorch** — Neural network construction, automatic differentiation, and optimization
* **NumPy** — Numerical operations and data processing
* **Matplotlib** — Scientific visualization
* **pyDOE** — Latin Hypercube Sampling for collocation-point generation

## Repository Structure

```text
pinn-internal-pipe-flow/
│
├── README.md
├── requirements.txt
├── pinn_pipe_flow.py
│
└── images/
    ├── radial_velocity.png
    ├── azimuthal_velocity.png
    ├── axial_velocity.png
    ├── pressure_contour.png
    ├── centerline_pressure.png
    └── friction_factor.png
```

*The structure above is a suggested organization for the repository. The script and image filenames should be adjusted to match the actual uploaded files.*

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/mreza-sotoudeh/pinn-internal-pipe-flow.git
cd pinn-internal-pipe-flow
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
numpy
matplotlib
pyDOE
```

### 3. Run the Simulation

```bash
python pinn_pipe_flow.py
```

The script trains the neural network and generates the velocity contours, pressure contour, centerline pressure plot, and friction-factor plot.

## Key Takeaways

* Explored the use of physics-informed learning for internal-flow simulation.
* Implemented a neural network with coordinate-based inputs and multiple physical outputs.
* Used automatic differentiation and collocation points to construct a physics-based training procedure.
* Investigated velocity and pressure fields through scientific visualization.
* Examined pressure-gradient-based friction-factor estimation.

## Limitations and Future Work

Potential improvements include:

* Enforcing the complete incompressible Navier–Stokes momentum equations in the loss function.
* Explicitly implementing the no-slip wall condition and appropriate constraints for all velocity components.
* Improving numerical stability near the pipe centerline, where cylindrical-coordinate expressions may contain terms proportional to \(1/r\).
* Validating the predicted velocity profile and pressure drop against analytical Hagen–Poiseuille solutions.
* Comparing the PINN predictions with a conventional CFD solution.
* Investigating loss weighting, alternative optimizers, and collocation-point refinement.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics, 3, 422–440.
3. White, F. M. (2006). *Viscous Fluid Flow.* McGraw-Hill.
4. Kundu, P. K., Cohen, I. M., & Dowling, D. R. (2016). *Fluid Mechanics.* Academic Press.

---

**Author:** Mohammadreza Sotoudeh
**Field:** Mechanical Engineering | Fluid Mechanics | Scientific Machine Learning

