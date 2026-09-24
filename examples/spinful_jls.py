"""Spin-1/2 S+P moving-frame JLS matrix with every allowed J block."""
import numpy as np
from lattice_scattering.finite_volume import row_matrix
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.symmetry import double_cover, little_group

frame = LatticeFrame(24, 1.0, (0, 0, 1))
sectors = ((0, 1), (1, 1))  # S_1/2 and P_1/2,P_3/2
# twice_J=1 contains both sector entries; twice_J=3 contains P only.
# Opposite orbital parity forbids off-diagonal S/P entries in the J=1/2 block.
blocks = {
    1: lambda _s: np.diag([1.0, 2.0]),
    3: lambda _s: np.array([[2.0]]),
}
group = double_cover(little_group(frame.d))
matrix = row_matrix(
    0.9, frame, [(0.3, 0.4)], sectors, blocks,
    group=group, irrep="G1", intrinsic_parity=-1,
)
error = np.max(np.abs(matrix - matrix.conj().T))
print(f"shape={matrix.shape} hermitian_error={error:.3e}")
print("eigenvalues=", np.linalg.eigvalsh(matrix))
