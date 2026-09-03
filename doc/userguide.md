# User Guide

## Introduction

MLIPOps is a collection of PyTorch modules for use in creating machine learning interatomic potentials (MLIPs).  It
is written in Python using PyTorch and Triton to do calculations.  This design has important benefits.

- Completely portable.  It works on any computer that PyTorch can run on.
- Fast performance on any hardware that Triton supports.
- Very easy to install.  It's pure Python with nothing that needs to be compiled in advance.
- Differentiable using standard PyTorch functions.
- Clean, simple code that is easy to maintain.
- All operations are compatible with torch.compile.

The operations it provides can roughly be divided into two categories.

- Low level operations are generally useful for implementing a wide range of potentials, both machine learning based and
  physics based.  Examples include neighbor lists, pairwise interactions, and periodic boundary conditions.
- High level operations implement specific physical interactions, such as Coulomb or dispersion.  They are implemented
  using the low level operations, which provide common infrastructure for the library.

## Installation

You can install MLIPOps from PyPI with the command

```
pip install mlipops
```

Alternatively, if you want to install from source, check out the code from https://github.com/openmm/mlipops, `cd` to
the root directory, and type

```
pip install .
```

## Low Level Operations

### Neighbor Lists

The `NeighborList` class finds pairs of particles that can interact with each other.  Most often (though not always)
this means ones within a cutoff distance of each other.  In a typical use, you simply provide the cutoff distance and
PyTorch device to the constructor.

```python
from mlipops import NeighborList
neighbor_list = NeighborList(cutoff=2.0, device='cuda')
```

You can then invoke it, providing a tensor of particle positions, and it returns the pairs within the cutoff distance.
An appropriate algorithm is chosen automatically based on the number of particles and the device.

```python
pairs = neighbor_list(positions)
```

`NeighborList` provides automatic caching of results.  If you pass the same tensor of positions to it twice in a row, it
will immediately return the cached neighbors without needing to recompute them.  In addition, you can optionally
specify a padding distance.

```python
neighbor_list = NeighborList(cutoff=1.0, padding=0.1, device='cuda')
```

In that case, it returns all pairs within `cutoff+padding`.  When you invoke it again, it checks whether any particle
has moved by more than `padding/2`.  If not, it can immediately return the cached neighbors since they are guaranteed to
include all pairs within `cutoff`.

An important consequence of caching is that you must never modify the inputs or outputs in place after calling the
`NeighborList`.  For example, if you call it once with a tensor of positions, modify the tensor in place to contain
different positions, and then pass it to the `NeighborList` again, it will incorrectly think the positions have not
changed.

`NeighborList` has other important features.  It supports periodic boundary conditions and batch processing (both
discussed below), as well as options for whether to include symmetric pairs and self interactions.  See the API
documentation for full details.

### Pairwise Interactions

The `Pairwise` class can be used to compute arbitrary pairwise interactions between particles.  The interaction can
depend on the distance and displacement vector between each pair of particles, as well as arbitrary parameters you
define.

To use it, first define a Python function to compute the interaction.  For example, the following function computes the
Coulomb energy $q_1 q_2/r$.

```python
def coulomb(pairs, r, deltas, charges):
    return charges[pairs[:,0]]*charges[pairs[:,1]]/r
````

It takes four arguments.  `pairs` lists the indices of interacting particles in the format returned by `NeighborList`.
`r` is the distance between each pair of interacting particles.  `deltas` is the displacement vector between each pair.
The final argument is an arbitrary object of any type you choose containing additional information.  In the above
example, it would be a 1D tensor containing the charge on each particle.

Next create a Pairwise object, passing your function to the constructor.

```python
from mlipops import Pairwise
pairwise = Pairwise(coulomb)
```

Finally evaluate it, passing in the particle positions, parameters, and pairs.  The result is automatically summed over
all pairs.

```python
energy = pairwise(positions, charges, pairs)
```

`Pairwise` has other important features.  It supports periodic boundary conditions and batch processing (both discussed
below), as well as options for excluding particular pairs or applying an additional cutoff.  See the API  documentation
for full details.

### Periodic Boundary Conditions

MLIPOps supports periodic boundary conditions with a triclinic box.  The box is specified by three box vectors, often
referred to as **a**, **b**, and **c**, stored as the rows of a 3x3 tensor.  In the common case of a rectangular box,
the diagonal elements contain the box  dimensions and the off-diagonal elements are zero.

To apply periodic boundary conditions, just pass the box vectors as an additional argument.

```python
pairs = neighbor_list(positions, box_vectors)
```

There also are functions you can call to apply periodic boundary conditions to displacements, such as
`periodic_displacements()` and `pairwise_displacements()`.  See the API documentation for details.

The box vectors must be chosen to satisfy certain requirements.  Roughly speaking, **a**, **b** , and **c** need to
"mostly" correspond to the x, y, and z axes respectively.  They must have the form **a** = ($a_x$, 0, 0), **b** =
($b_x$, $b_y$, 0), and **c** = ($c_x$, $c_y$, $c_z$).  It is always possible to put them into this form by rotating the
system until **a** is parallel to x and **b** lies in the xy plane.  In addition, they must obey certain constraints:

- $a_x$, $b_y$, and $c_z$ are all positive
- $a_x \ge 2 |b_x|$
- $a_x \ge 2 |c_x|$
- $b_y \ge 2 |c_y|$

This requires the box vectors to be specified in a particular reduced form.  By forming combinations of box vectors (a
process known as "lattice reduction"), it is always possible to put them in this form without changing the periodic
system they represent.

It is your responsibility to ensure that the box vectors you provide satisfy these requirements.  MLIPOps does not check
them, because doing so would require extra communication between GPU and CPU and hurt performance.  If you provide box
vectors that violate these requirements, they can lead to incorrect results.

Another important requirement applies to interactions that involve a cutoff distance between particles.  When used with
periodic boundary conditions, all dimensions of the periodic box ($a_x$, $b_y$, and $c_z$) must be at least twice the
cutoff distance.  Otherwise, two copies of a single particle could simultaneously be within the cutoff distance of
another particle.  If this requirement is violated, it can lead to incorrect results.  MLIPOps does not attempt to
verify that it is satisfied.  It is your responsibility to make sure the box vectors you provide satisfy it.

### Batch Processing

MLIPOps supports batch processing.  Instead of computing an operation for a single system, it computes it for many
systems at once.  When the individual systems are small, this can be far more efficient than processing them one at a
time.

To perform batch processing, combine all the particles from all the systems into a single flat list.  That is, there is
a single tensor containing all the positions, a single tensor containing all the charges for Coulomb interactions, etc.
Then pass a `batch` argument to the operation containing the index of the system each particle belongs to.  For example,

```python
pairs = neighbor_list(positions, batch=batch)
```

It will perform the calculation in a way that keeps the systems separate.  For example, the neighbor list above will
only return pairs of particles that are both in the same system.  In addition, operations that return an energy will
produce a tensor containing a separate energy for each system.

The `batch` argument must be in sorted order, and every system must contain at least one particle.  In other words,
`batch` should contain one or more 0's, followed by one or more 1's, followed by one or more 2's, and so on.  If these
requirements are violated, it can lead to incorrect results.

## High Level Operations

High level operations implement specific physical interactions.  They are built using the low level operations described
above.  They are typically combined with machine learning to help a potential achieve better accuracy or
transferability.

### Units

MLIPOps lets you use whatever system of units you want, but to compute physical interactions it needs to know what units
you are using.  You tell it that by providing the values of two physical constants in your chosen unit system: Coulomb's
constant $\frac{1}{4 \pi \epsilon_0}$ and the Bohr radius $a_0$.  The values of these constants depend on the units used
to measure distance and energy.  Here are their values for a variety of common sets of units.

| Energy Unit | Distance Unit | Coulomb's Constant | Bohr Radius |
|-------------|---------------|--------------------|-------------|
| kJ/mol      | nm            | 138.935457         | 0.052917721 |
| kJ/mol      | Å             | 1389.35457         | 0.52917721  |
| kcal/mol    | nm            | 33.2063713         | 0.052917721 |
| kcal/mol    | Å             | 332.063713         | 0.52917721  |
| eV          | nm            | 1.43996454         | 0.052917721 |
| eV          | Å             | 14.3996454         | 0.52917721  |
| hartree     | bohr          | 1.0                | 1.0         |

### Coulomb Interactions

MLIPOps offers a variety of methods for computing Coulomb interactions between point charges and dipoles.  They differ
in how they handle long range interactions and periodic boundary conditions.

- **No Cutoff**.  All interactions are included regardless of how far apart they are.  This method is only applicable to
  non-periodic systems.  The cost scales as $N^2$ in the number of particles, making this mainly useful for smaller
  systems.
- **Reaction Field**.  Interactions beyond a cutoff distance are ignored.  The energy is smoothly reduced to zero at the
  cutoff distance using the reaction field approximation, which assumes everything beyond the cutoff is filled with a
  uniform dielectric.  This method can be applied to both periodic and non-periodic systems and is fast to compute in
  all cases.  Unlike the other methods, only charges are supported, not dipoles.
- **Ewald Summation**.  Ewald summation is used to compute the full set of infinite interactions in a periodic system.
- **Particle Mesh Ewald**.  Particle Mesh Ewald (PME) is used to compute the full set of infinite interactions in a
  periodic system.  This method has better scaling with system size than conventional Ewald summation, making it much
  faster for large systems.  Unlike the other methods, only first derivatives are supported.  This makes it primarily
  useful for inference rather than training.

Here is an example of computing Coulomb interactions with the reaction field method.

```python
from mlipops import NeighborList, CoulombRF
neighbor_list = NeighborList(cutoff=1.2, device='cuda')
coulomb = CoulombRF(neighbor_list, None, 138.935457)
energy = coulomb(positions, charges, box_vectors)
```

We begin by creating the `NeighborList` that will be used to find neighbors.  This sets the cutoff distance, the device
on which to perform calculations, and whether to add padding to improve caching of results.

Next we create a `CoulombRF` object to perform the calculation.  We provide it with the `NeighborList` and the value of
Coulomb's constant.  We can also specify a list of excluded pairs whose interaction should be omitted.  In this case we
do not use exclusions, so we pass `None` for that argument.

Finally we compute the energy by invoking the `CoulombRF` object, passing in the positions, charges, and periodic box
vectors.

Here is a more complicated example.  In this case we use Ewald summation, include dipoles as well as charges, and
compute the energy for a batch of systems.

```python
from mlipops import NeighborList, CoulombEwald
neighbor_list = NeighborList(cutoff=0.9, device='cuda')
coulomb = CoulombEwald(neighbor_list, None, 9, 9, 9, 5.0, 138.935457, max_multipole='dipole')
energy = coulomb(positions, charges, box_vectors, dipoles=dipoles, batch=batch)
```

The `CoulombEwald` class requires a few extra arguments to the constructor: the number of reciprocal space wave vectors
to include  along each axis, and the coefficient of the `erf()` function used to separate the energy into real space and
reciprocal space parts.  See the API documentation for details.  We also specify `max_multipole='dipole'` to tell it
this object will be used to compute energies of dipoles.

Invoking the object is exactly the same as before, but we include two extra arguments.  `dipoles` contains the dipole
moment of each particle.  `batch` specifies which system each particle belongs to, as described above.  The returned
value is a tensor whose length equals the number of systems, containing the energy of each system.

### Charge Equilibration

Charge equilibration is a method of determining partial charges for atoms.  It models each atom as a Gaussian charge
distribution.  You specify three parameters for each atom: its electronegativity $\chi_i$, which describes its innate
affinity for electrons; its hardness $J_{ii}$, which describes its resistance to changes in charge; and its radius
$\alpha_i$, which is the width of the Gaussian charge distribution.  We then solve for the charges $q$ that minimize the
energy

$E = E_{coul}(r, q, \alpha) + \sum_{i} \left[ \chi_i q_i + \frac{1}{2} \left( J_{ii} + \frac{\sqrt{2}}{\sqrt{\pi} \alpha_i} \right) q_i^2 \right]$

subject to a constraint on the total charge, where $E_{coul}(r, q, \alpha)$ is the energy of the Coulomb interaction.

To use charge equilibration, first create an object to compute the Coulomb interaction.  This determines the method used
to handle long range interactions (no cutoff, reaction field, or Ewald summation) and its parameters (e.g. the number of
wave vectors for Ewald summation or the `NeighborList` used to identify interacting pairs).

```python
from mlipops import CoulombNC, ChargeEquilibration
coulomb = CoulombNC(None, 138.935)
eq = ChargeEquilibration(coulomb)
```

You can then invoke it to compute the charges on atoms.

```python
charges = eq(positions, electronegativity, hardness, radius, total_charge=0)
```

In this example we specified a single value for the total charge of the system.  That works for a single isolated
molecule, but generally should not be used for systems of multiple molecules, since it leads to unphysical charge
transfer and molecules with fractional charges.  Instead you can impose a separate constraint on the total charge of
each molecule.  To do this, provide a list of the atoms making up each molecule and the total charge of each molecule.

```python
molecules = [(mol1_atoms, mol1_charge), (mol2_atoms, mol2_charge)]
charges = eq(positions, electronegativity, hardness, radius, molecules=molecules)
```

This produces more accurate results, but it requires you to know in advance how the atoms are grouped into molecules,
and what the total charge of each molecule should be.

You can specify an externally imposed electric field that should influence the charge distribution.  To do this, pass
an extra argument containing the external potential at the location of each atom.  A common use for this feature is in
ML/MM simulations, where charges in the MM region should polarize the ML region.  In this case, you can use
`compute_potential()` to determine the potential induced by the MM charges.

```python
potential = coulomb.compute_potential(ml_positions, mm_positions, mm_charges)
ml_charges = eq(ml_positions, electronegativity, hardness, radius, total_charge=0, potential=potential)
```

Another option is to specify a default charge $q^0_i$ for every atom.

```python
charges = eq(positions, electronegativity, hardness, radius, total_charge=0, default_charge=default_charge)
```

In that case, the energy function is modified to

$E = E_{coul}(r, q, \alpha) + \sum_{i} \left[ \chi_i (q_i - q^0_i) + \frac{1}{2} \left( J_{ii} + \frac{\sqrt{2}}{\sqrt{\pi} \alpha_i} \right) (q_i - q^0_i)^2 \right]$

In some cases, using nonzero default charges (for example, the formal oxidation state of each atom) can lead to more
accurate results.

### Dispersion

Dispersion energy can be computed using the DFT-D3(BJ) model.  Using it is similar to the Coulomb examples shown above,
but with a few differences.

```python
from mlipops import NeighborList, DFTD3, get_covalent_radii
neighbor_list = NeighborList(device='cuda')
d3 = DFTD3(neighbor_list, 0.78981345, 0.49484001, 5.73083694, 138.935, 0.052917721)
radii = get_covalent_radii(numbers, 0.052917721)
energy = d3(positions, numbers, radii)
```

In this example, we did not specify a cutoff distance for the `NeighborList`.  That means all pairs will be included
regardless of how far apart they are.  This choice is appropriate for small molecules, but the cost scales as $N^2$ in
the number of particles.  For larger systems you will generally use a cutoff.

The last two arguments to the `DFTD3` constructor are the values of Coulomb's constant and the Bohr radius in our chosen
unit system.  In addition, we specify the values of three parameters of the DFT-D3(BJ) model called `s8`, `a1`, and
`a2`.  The values of these parameters are tuned for each DFT functional the model is combined with.  When used with a
machine learning potential, you should generally use the parameters appropriate to the functional used to generate the
training data.

The dispersion calculation depends on the atomic numbers of the interacting atoms.  In this example, we assume `numbers`
is a tensor containing the atomic numbers.  It also depends on the covalent radii of the atoms.  Given the atomic
numbers, you can use the function `get_covalent_radii()` to look up the corresponding radii.

### Nuclear Repulsion

The Ziegler-Biersack-Littmark (ZBL) potential models the screened nuclear repulsion between atoms.  It is frequently
combined with machine learning potentials to ensure realistic behavior when atoms come very close together, a situation
for which there often is no training data.  Here is an example of using it.

```python
from mlipops import NeighborList, ZBL, get_covalent_radii
radii = get_covalent_radii(numbers, 0.052917721)
neighbor_list = NeighborList(2*radii.max(), device='cuda')
zbl = ZBL(neighbor_list, 138.935, 0.052917721)
energy = zbl(positions, numbers, radii, box_vectors)
```

Similar to `DFTD3`, we pass atomic numbers and covalent radii when evaluating the potential.  In this case the reasons
are slightly different.  As originally published, the ZBL potential depends only on atomic numbers.  However, it was
parameterized based on scattering data and does not give an accurate description of atoms that are covalently bonded to
each other.  For that reason, it often is restricted to very short distances.  If you provide covalent radii, the
potential is multiplied by a cutoff function that smoothly reduces it to zero when the distance between two atoms equals
the sum of their covalent radii.  This usually is a very short distance, just a few angstroms.  When creating the
`NeighborList`, we set the cutoff distance to twice the maximum covalent radius.  Any atoms further apart than that are
guaranteed not to interact and can safely be omitted.