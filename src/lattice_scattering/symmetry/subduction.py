"""Character projection for a supplied, consistently ordered finite group."""
from numbers import Integral
import numpy as np


def isotypic_subspace(representations,characters,*,irrep_dimension,max_dimension=256,max_group_order=128):
    """Return projector/basis for all copies of one supplied irrep.

    Caller supplies a unitary group representation and irreducible character
    in identical element order. This checks numerical consistency, not the
    group multiplication law or correctness of a physical irrep assignment.
    Columns span all rows and copies together; no row/occurrence labels are
    inferred from eigenvector order. Missing irreps return an empty basis.
    """
    for value in (irrep_dimension,max_dimension,max_group_order):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
            raise ValueError('positive integer dimensions and budgets required')
    if not 0<len(representations)<=max_group_order:
        raise ValueError('group order exceeds budget or is empty')
    matrices=np.asarray(representations,dtype=complex)
    if matrices.ndim!=3 or matrices.shape[1]!=matrices.shape[2] or not 0<matrices.shape[1]<=max_dimension or not np.all(np.isfinite(matrices)):
        raise ValueError('finite square representation matrices within dimension budget required')
    order,dimension,_=matrices.shape
    chars=np.asarray(characters,dtype=complex)
    if chars.shape!=(order,) or not np.all(np.isfinite(chars)):
        raise ValueError('one finite character per group element required')
    tolerance=1e-10
    if not np.isclose(np.vdot(chars,chars).real/order,1,rtol=0,atol=tolerance) or np.any(abs(chars)>irrep_dimension+tolerance):
        raise ValueError('characters fail irreducible normalization/dimension checks')
    identity=np.eye(dimension)
    if any(not np.allclose(u.conj().T@u,identity,rtol=0,atol=tolerance) for u in matrices):
        raise ValueError('unitary representation matrices required')
    projector=irrep_dimension/order*np.einsum('g,gij->ij',chars.conj(),matrices)
    if not np.allclose(projector,projector.conj().T,rtol=0,atol=tolerance) or not np.allclose(projector@projector,projector,rtol=0,atol=tolerance):
        raise ValueError('character sum is not an orthogonal projector')
    if any(not np.allclose(projector@u,u@projector,rtol=0,atol=tolerance) for u in matrices):
        raise ValueError('projector is not invariant under the supplied representation')
    values,vectors=np.linalg.eigh((projector+projector.conj().T)/2)
    rank=int(np.count_nonzero(values>.5))
    if rank%irrep_dimension:
        raise ValueError('projected rank is not a multiple of irrep dimension')
    basis=vectors[:,values>.5]
    projector.setflags(write=False);basis.setflags(write=False)
    return {'projector':projector,'basis':basis,'rank':rank,'multiplicity':rank//irrep_dimension,
            'irrep_dimension':int(irrep_dimension),'basis_labels':None}


def row_subduction(representations,irrep_matrices,*,max_dimension=256,max_group_order=128):
    """Aligned row/copy bases from P_r0 = d/G sum conjugate(D_r0(g)) U(g).

    Copy indices are numerical basis indices, not inferred J/ell/S labels.
    Eigenvectors in row zero may rotate within the multiplicity space; the
    same copy rotation is carried to every row. Caller owns group conventions.
    """
    for value in (max_dimension,max_group_order):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
            raise ValueError('positive integer dimensions and budgets required')
    if not 0<len(irrep_matrices)<=max_group_order:
        raise ValueError('irrep group order exceeds budget or is empty')
    carrier=np.asarray(irrep_matrices,dtype=complex)
    if carrier.ndim!=3 or carrier.shape[1]!=carrier.shape[2] or not carrier.shape[1] or not np.all(np.isfinite(carrier)):
        raise ValueError('finite square irrep matrices required')
    dimension=carrier.shape[1]
    if dimension>max_dimension:
        raise ValueError('irrep dimension exceeds budget')
    if any(not np.allclose(d.conj().T@d,np.eye(dimension),rtol=0,atol=1e-10) for d in carrier):
        raise ValueError('unitary irrep matrices required')
    result=isotypic_subspace(representations,np.trace(carrier,axis1=1,axis2=2),
                            irrep_dimension=dimension,max_dimension=max_dimension,max_group_order=max_group_order)
    matrices=np.asarray(representations,dtype=complex)
    if len(carrier)!=len(matrices):
        raise ValueError('group element counts differ')
    factor=dimension/len(matrices)
    p00=factor*np.einsum('g,gij->ij',carrier[:,0,0].conj(),matrices)
    if not np.allclose(p00,p00.conj().T,rtol=0,atol=1e-10) or not np.allclose(p00@p00,p00,rtol=0,atol=1e-10):
        raise ValueError('row-zero matrix unit is not an orthogonal projector')
    values,vectors=np.linalg.eigh((p00+p00.conj().T)/2)
    initial=vectors[:,values>.5]
    copies=result['multiplicity']
    if initial.shape[1]!=copies:
        raise ValueError('row-zero multiplicity differs from character projection')
    rows=[]
    for row in range(dimension):
        unit=factor*np.einsum('g,gij->ij',carrier[:,row,0].conj(),matrices)
        rows.append(unit@initial)
    basis=np.concatenate(rows,axis=1)
    if not np.allclose(basis.conj().T@basis,np.eye(dimension*copies),rtol=0,atol=1e-10) or not np.allclose(basis@basis.conj().T,result['projector'],rtol=0,atol=1e-10):
        raise ValueError('row bases do not resolve the isotypic subspace')
    for u,d in zip(matrices,carrier):
        if not np.allclose(u@basis,basis@np.kron(d,np.eye(copies)),rtol=0,atol=1e-10):
            raise ValueError('row bases do not intertwine the supplied representations')
    for row in rows:row.setflags(write=False)
    basis.setflags(write=False)
    result.update(basis=basis,row_bases=tuple(rows),basis_labels=tuple(
        {'row':row,'copy_index':copy} for row in range(dimension) for copy in range(copies)))
    return result
