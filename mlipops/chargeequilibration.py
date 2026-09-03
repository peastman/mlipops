import torch
import math
from .coulombnc import CoulombNC
from .coulombrf import CoulombRF
from .coulombewald import CoulombEwald
from .minres import minres
from .utils import pairwise_displacements, batch_pairwise_displacements


class ChargeEquilibration(torch.nn.Module):
    """Compute atomic partial charges with charge equilibration.

    This class implements the charge equilibration method as described in http://dx.doi.org/10.1103/PhysRevB.92.045131.
    It models atoms as Gaussian charge distributions.  You provide three parameters for each atom: electronegativity,
    which describes its innate affinity for electrons; hardness, which describes its resistance to changes in charge;
    and radius, which is the width of the Gaussian charge distribution.  It solves for the charges that minimize an
    energy function subject to constraints on the total charge.

    In the simplest version, you provide a single number for the total charge of the system, and it constrains the
    atomic partial charges to add up to the correct value.  This works well for single isolated molecules, but cannot
    accurately describe systems of multiple molecules.  In particular, it unrealistically predicts fractional charges
    for molecules.  Instead you can provide a list of the atoms that make up each molecule and the total charge of each
    one.  It then solves the equations subject to a separate constraint for each molecule.  This produces more realistic
    results, but requires that you know in advance how the atoms are divided into molecules and how the charge is
    divided among them.

    You can optionally provide an external electric potential that should be used to polarize the atoms.  This might
    be computed from a uniform external field, or by calling compute_potential() on a Coulomb calculation object to get
    the potential resulting from a set of external charges.

    You can optionally specify a default charge for each atom.  The energy function is then modified as described in
    https://doi.org/10.1021/jz3008485 to bias each atom towards its default charge.  In some cases, using nonzero
    default charges (for example, the formal oxidation state of each atom) can lead to more accurate results.

    This class offers a choice of method for solving the system of equations.  By default it uses torch.linalg.solve(),
    which implements a direct algorithm.  It is accurate and generally robust, but it can be slow, especially for large
    systems.  Alternatively you can choose MINRES, an efficient iterative algorithm.  It can be much faster in some
    cases, but this comes at the cost of somewhat lower accuracy.
    """

    def __init__(self, coulomb: CoulombNC | CoulombRF | CoulombEwald):
        """Create an object for performing charge equilibration.

        Parameters
        ----------
        coulomb: CoulombNC | CoulombRF | CoulombEwald
            the object used to compute Coulomb interactions between atoms.  This determines what method is used for
            handling long range interactions and the neighbor list used for identifying interacting pairs.
        """
        super().__init__()
        self.coulomb = coulomb

    def forward(self, positions: torch.Tensor, electronegativity: torch.Tensor, hardness: torch.Tensor,
                radius: torch.Tensor, total_charge: float | torch.Tensor | None = None, molecules: list | None = None,
                box_vectors: torch.Tensor | None = None, potential: torch.Tensor | None = None,
                default_charge: torch.Tensor | None = None, batch: torch.Tensor | None = None,
                solver: str = 'direct') -> torch.Tensor:
        """Perform charge equilibration to compute atomic partial charges.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        electronegativity: torch.Tensor
            a Tensor of shape (n_particles,) containing the electronegativity ($\\chi$) of each particle
        hardness: torch.Tensor,
            a Tensor of shape (n_particles,) containing the hardness ($J_{ii}$) of each particle
        radius: torch.Tensor
            a Tensor of shape (n_particles,) containing the radius ($\\alpha$) of each particle
        total_charge: float | torch.Tensor | None
            the total charge of the system.  If batch is None, this should be a float or scalar Tensor.  If batch is not
            None, this should be a Tensor of shape (n_systems,) containing the charge of each system.  You must specify
            either total_charge or molecules, but not both.
        molecules: list | None
            the list of molecules.  Each element should be a tuple with two elements.  The first element is a Tensor
            containing the indices of the particles that belong to the molecule.  The second element is a float with the
            total charge of the molecule.  You must specify either total_charge or molecules, but not both.
        box_vectors: torch.Tensor | None
            if batch is None, a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If batch is
            not None, a Tensor of shape (n_systems, 3, 3) containing the box vectors for each system.  If None, periodic
            boundary conditions are not used.
        potential: torch.Tensor
            a Tensor of shape (n_particles,) containing the external electric potential at the location of each particle
        default_charge: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the default charge of every particle.  If None, all default
            charges are 0.
        solver: str
            the method to use for solving the system of equations.  Options are 'direct' (use torch.linalg.solve() to
            directly compute the result) and 'minres' (an iterative solver that tends to be faster, especially for large
            systems, at the cost of slightly lower accuracy).
        batch: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the index of the system each particle belongs to.  This must be
            sorted in ascending order, and every system must contain at least one particle.  If None, the calculation
            is performed for a single system instead of a batch of systems.

        Returns
        -------
        torch.Tensor:
            a Tensor of shape (n_particles,) containing the charge of each particle
        """
        # Build the interaction matrix.

        num_systems = 1 if batch is None else batch[-1]+1
        if isinstance(self.coulomb, CoulombNC):
            if box_vectors is not None:
                raise ValueError('Cannot use periodic boundary conditions with CoulombNC')
            interaction = self._compute_interactions_nc(positions, hardness, radius, batch)
        elif isinstance(self.coulomb, CoulombRF):
            interaction = self._compute_interactions_rf(positions, hardness, radius, box_vectors, batch)
        elif isinstance(self.coulomb, CoulombEwald):
            if box_vectors is None:
                raise ValueError('Must specify box_vectors for CoulombEwald')
            interaction = self._compute_interactions_ewald(positions, hardness, radius, box_vectors, batch, num_systems)
        else:
            raise ValueError('coulomb must be a CoulombNC, CoulombRF, or CoulombEwald')

        # Build the matrix describing constraints and the list of total charges.

        device = positions.device
        n = positions.shape[0]
        if total_charge is not None:
            if molecules is not None:
                raise ValueError('total_charge and molecules were both specified')
            if batch is not None:
                constraint = torch.zeros((n, num_systems), dtype=torch.float32, device=device)
                constraint[torch.arange(n, device=device), batch] = 1
                zeros = torch.zeros((num_systems, num_systems), dtype=torch.float32, device=device)
                mol_charges = total_charge
            else:
                constraint = torch.ones((n, 1), dtype=torch.float32, device=device)
                zeros = torch.zeros((1, 1), dtype=torch.float32, device=device)
                mol_charges = torch.tensor([total_charge], dtype=torch.float32, device=device)
        elif molecules is not None:
            m = len(molecules)
            constraint = torch.zeros((n, m), dtype=torch.float32, device=device)
            mol_charges = []
            for i, (indices, charge) in enumerate(molecules):
                constraint[indices, i] = 1
                mol_charges.append(charge)
            zeros = torch.zeros((m, m), dtype=torch.float32, device=device)
            mol_charges = torch.tensor(mol_charges, dtype=torch.float32, device=device)
        else:
            raise ValueError('Neither total_charge nor molecules was specified')

        # Build the tensors representing the system of equations.

        matrix = torch.cat([torch.cat([interaction, constraint], dim=1),
                            torch.cat([constraint.T, zeros], dim=1)])
        rhs = electronegativity
        if potential is not None:
            rhs = rhs+potential
        if default_charge is not None:
            rhs = rhs-interaction.diag()*default_charge
        x = torch.cat([-rhs, mol_charges])

        # Solve the system of equations.

        if solver == 'direct':
            return torch.linalg.solve(matrix, x)[:n]
        if solver == 'minres':
            return minres(matrix, x, tol=1e-7)[:n]
        raise ValueError(f'Illegal value for solver: {solver}')

    def _compute_interactions_nc(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor,
                                 batch: torch.Tensor | None) -> torch.Tensor:
        """Build the interaction matrix when using CoulombNC."""
        n = positions.shape[0]
        interactions = torch.zeros((n, n), dtype=torch.float32, device=positions.device)
        pairs = self.coulomb.neighbor_list(positions, None, batch)
        if batch is None:
            delta = pairwise_displacements(positions, pairs, None)
        else:
            delta = batch_pairwise_displacements(positions, pairs, batch, None)
        distance = torch.linalg.vector_norm(delta, dim=1)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2[pairs[:,0]] + radius2[pairs[:,1]])
        values = torch.erf(gamma*distance)/distance
        interactions[pairs[:,0], pairs[:,1]] = values
        interactions[pairs[:,1], pairs[:,0]] = values
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           interactions)

    def _compute_interactions_rf(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor,
                                 box_vectors: torch.Tensor | None, batch: torch.Tensor | None) -> torch.Tensor:
        """Build the interaction matrix when using CoulombRF."""
        n = positions.shape[0]
        interactions = torch.zeros((n, n), dtype=torch.float32, device=positions.device)
        pairs = self.coulomb.neighbor_list(positions, box_vectors, batch)
        if batch is None:
            delta = pairwise_displacements(positions, pairs, box_vectors)
        else:
            delta = batch_pairwise_displacements(positions, pairs, batch, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2[pairs[:,0]] + radius2[pairs[:,1]])
        k = self.coulomb.pairwise.computation.k
        c = self.coulomb.pairwise.computation.c
        values = torch.erf(gamma*distance) * (1/distance + k*distance**2 - c)
        interactions[pairs[:,0], pairs[:,1]] = values
        interactions[pairs[:,1], pairs[:,0]] = values
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           interactions)

    def _compute_interactions_ewald(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor,
                                    box_vectors: torch.Tensor | None, batch: torch.Tensor | None, num_systems: int) -> torch.Tensor:
        """Build the interaction matrix when using CoulombEwald."""
        n = positions.shape[0]
        interactions = torch.zeros((n, n), dtype=torch.float32, device=positions.device)

        # Compute direct space interactions.

        pairs = self.coulomb.neighbor_list(positions, box_vectors, batch)
        if batch is None:
            delta = pairwise_displacements(positions, pairs, box_vectors)
        else:
            delta = batch_pairwise_displacements(positions, pairs, batch, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2[pairs[:,0]] + radius2[pairs[:,1]])
        values = (torch.erf(gamma*distance) - torch.erf(self.coulomb.alpha*distance))/distance
        interactions[pairs[:,0], pairs[:,1]] = values
        interactions[pairs[:,1], pairs[:,0]] = values

        # Compute reciprocal space interactions.

        recip_box_vectors = torch.linalg.inv(box_vectors)
        if batch is None:
            k = self.coulomb.wave_indices@(2*torch.pi*recip_box_vectors.T)
            phase = k@positions.T
            cos = phase.cos()
            sin = phase.sin()
            k2 = (k*k).sum(dim=1)
            ak = torch.exp(self.coulomb._exp_coeff*k2)/k2
            interactions += torch.einsum('i,ij,ik->jk', ak, cos, cos) + torch.einsum('i,ij,ik->jk', ak, sin, sin)
            interactions *= 4*torch.pi*recip_box_vectors.diag().prod()
        else:
            k = self.coulomb.wave_indices.unsqueeze(0)@(2*torch.pi*recip_box_vectors.transpose(1, 2))
            phase = torch.einsum('ijk,ik->ji', k[batch], positions)
            cos = phase.cos()
            sin = phase.sin()
            k2 = (k*k).sum(dim=2)
            ak = (torch.exp(self.coulomb._exp_coeff*k2)/k2)[batch]
            interactions += torch.einsum('ji,ij,ik->jk', ak, cos, cos) + torch.einsum('ji,ij,ik->jk', ak, sin, sin)
            box_scale = torch.diagonal(recip_box_vectors, dim1=1, dim2=2).prod(dim=1)[batch]
            interactions = torch.where(batch == batch.unsqueeze(1), (4*torch.pi)*interactions*box_scale, 0.0)
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           interactions)
