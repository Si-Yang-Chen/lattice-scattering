"""One-channel S-wave forward scan with an explicit K-matrix convention."""
import numpy as np
from lattice_scattering.amplitudes import ConstantK, effective_inverse_k
from lattice_scattering.finite_volume import quantization_roots, row_matrix
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.symmetry import double_cover, little_group

frame = LatticeFrame(24, 1.0, (0, 0, 0))
masses = [(0.3, 0.3)]
model = ConstantK(matrix=[[2.0]])
sectors = ((0, 0),)
group = double_cover(little_group(frame.d))


def reduced_inverse(s_at2):
    # Tested mapping for this S-wave channel-basis convention only.
    inverse = effective_inverse_k(model, masses, "simple", s_at2)
    return np.sqrt(s_at2) * frame.length_at / (4.0 * np.pi) * inverse


j_blocks = {0: reduced_inverse}
roots = quantization_roots(
    (0.61, 0.95), frame, masses, sectors, j_blocks,
    group=group, irrep="A1g", weighting="threshold", samples=60,
)
for energy in roots:
    matrix = row_matrix(
        energy, frame, masses, sectors, j_blocks,
        group=group, irrep="A1g", weighting="threshold",
    )
    print(f"energy_lab_at={energy:.12f} det={np.linalg.det(matrix).real:.3e}")
