# CMM

[![Build Status](https://github.com/heindelj/CMM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/heindelj/CMM.jl/actions/workflows/CI.yml?query=branch%3Amain)

# About
The Completely Multipolar Model (CMM) is a model of intermolecular forces designed to reproduce the five terms of an energy decomposition analysis: Pauli repulsion, electrostatics, polarization, charge transfer, and dispersion. Here are some unique apects of CMM compared to other force fields:
- Systematically improvable through inclusion of higher-order multipoles for any term in the model
- Explicit charge transfer between molecules
- Quantitative model of many-body charge transfer
- Anisotropic models of both Pauli repulsion and charge transfer enabled by sharing anisotropy with electrical multipoles
- Coupling between intermolecular potentials and bonding potential via a field-dependent Morse potential
- Predicts all terms from EDA separately rather than combining terms in an *ad hoc* manner


# Install
Currently, we do not provide CMM as a general Julia package, but might in the future. For now, please install by cloning this repository and activating the Julia environment. Here are some step-by-step commands to clone the repository and then run a calculation on the water dimer in the Julia REPL. Various cluster geometries are available in the assets folder and some utility functions are provided for using the model. Please contact me with any questions or difficulties.

`> git clone https://github.com/heindelj/CMM.jl.git`

`> cd CMM.jl`

`> julia`

`> import Pkg`

`> Pkg.activate(".")`

`> using CMM`

The first time you do the previous command, the dependencies of the project will be installed. If that succeeds, then the installation succeeded and you should be able to run the following commands in the same Julia REPL session.

# Quick Start

`> header, labels, geoms = read_xyz("assets/xyz/w2_wb97xv_qzvppd_rigid_monomers.xyz")`

`> ff = build_cmm_model()`

`> evaluate!(geoms[1] / .529177, labels[1], [[i, i+1, i+2] for i in 1:3:length(labels[1])], ff)`

`> ff.results.energies`

- The `evaluate!` command takes four arguments:
	- The xyz coordinates of the molecule in bohr (hence dividing by 0.529177)
	- The atomic labels associated with each atom
	- The connectivity of the molecules as `Vector{Vector{Int}}` (i.e. for the water dimer, this would be [[1,2,3], [4,5,6]])
	- A CMM_FF object which stores the results of the calculation and the intermediate values
- The command will evaluate the energy of the system and store the results in `ff.results.energies`
	- Notice that the energies object contains more than just the EDA terms described above as some of the terms have multiple contributions.
	- Another nuance is that the `Polarization` energy reported in `ff.results.energies` is not the actual polarization energy. This is because, just like in EDA, the charge transfer and polarization energies are coupled. The calculation you just did occurred on the physical PES, which includes explicit charge transfer. To get the actual polarization energy, we need to compute the energy on the POL surface as follows.

`> ff_pol = build_cmm_model()`

`> evaluate!(geoms[1] / .529177, labels[1], [[i, i+1, i+2] for i in 1:3:length(labels[1])], ff_pol)`

`> ff_pol.results.energies`


`> ff_pol = build_cmm_model_POL()`

`> evaluate!(geoms[1] / .529177, labels[1], [[i, i+1, i+2] for i in 1:3:length(labels[1])], ff_pol)`

`> ff_pol.results.energies`

- The polarization energy reported in `ff_pol.results.energies` is the actual polarization energy predicted by the model
- The correct charge transfer energy predicted by the model is: `ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])`
	- That is, the charge transfer includes the change in polarization energy induced by explicitly moving charge between molecules

`opt_energy, opt_geom = optimize_xyz_by_fd(geoms[1] / .529177, labels[1], [[i,i+1,i+2] for i in 1:3:length(geoms[1])], ff)`

- The above command will optimized a structure by finite difference returning the total optimized energy, `opt_energy`, and optimized structure, `opt_geom`.

- The forces are roughly 50% complete in this package. The complete forces will be available around the same time the condensed phase implementation is completed.

# Status
Currently, parameters are available for water and the halide anions and alkali cations. This code is intended for modelling clusters and for the development of new parameterizations. We are in the process of writing an implementation for condensed phase periodic boundary condition simulations. Also, I am planning to rewrite many aspects of this package to make it signficantly faster as well as to finish the implementation of forces. If you are interested in contributing or making a parameterization for a new set of molecules, please email me and I will be happy to assist or give some advice. I have a lot of tools not in this package to aid in fitting the various terms which I am hoping to organize into something that can be generically useful when constructing accurate potential energy surfaces.