# Exact selective-\( J \) projection in the JLS kernel

`selected_j_sectors` lets a caller retain a chosen subset of the channel/LS
incidences that participate in each `twice_J` block. It removes omitted
total-\( J \) content from the finite-volume subspace itself. It does not
approximate omitted waves with zero or very large inverse amplitudes.

## Input and block layout

The Python interfaces `assemble_inverse`, `row_matrix`, and
`quantization_roots` accept a mapping

```python
selected_j_sectors = {
    twice_J: ((channel_index, ell, twice_S), ...),
}
```

`channel_index` is zero-based. `twice_S` and `twice_J` are doubled angular
momenta. Every listed `(channel_index, ell, twice_S)` must occur in that
channel's declared sector list and satisfy the triangle rule for the key's
`twice_J`. A sector may be retained for one allowed `J` and omitted for
another. Omitting a `twice_J` key omits that entire block. An empty mapping or
an empty incidence list is invalid; omit a key to omit its `J` block.

For each retained `twice_J`, provide exactly one real symmetric amplitude
matrix whose order equals the number of selected incidences for that key.
Rows and columns are canonicalized to channel-major order, then to each
channel's declared sector order. The order of keys or incidences in the
selection mapping does not define amplitude-matrix order. Blocks for
unselected `J` values are extra and are rejected, as are missing blocks,
wrong dimensions, duplicate incidences, unknown sectors, and triangle-rule
violations.

In the CLI JSON, use string keys and arrays of triples. This one-channel S+P
example keeps `S31` and `P33` while excluding `P31` (for a spin-1/2 S+P
channel, `twice_S=1`; isospin in the spectroscopic labels is external to this
API):

```json
{
  "sectors": [[0, 1], [1, 1]],
  "selected_j_sectors": {
    "1": [[0, 0, 1]],
    "3": [[0, 1, 1]]
  },
  "j_blocks": {
    "1": [[1.2]],
    "3": [[0.8]]
  }
}
```

Here the full declared basis has `S31` at `twice_J=1` and both `P31` at
`twice_J=1` and `P33` at `twice_J=3`. The active `J=1` block is one-dimensional
and contains only `S31`; the active `J=3` block contains only `P33`. The
`j_blocks` values above are schema examples, not a fitted or validated
physical amplitude.

## Projected irrep row

Let `U_(c,ell,S,J)` be the orthonormal coupled-angular-momentum basis embedded
in the declared channel/LS carrier. The active projector is the direct sum,
over each channel and LS sector, of `U U†` for the selected `J` incidences in
that sector. It has no component along an omitted `J` subspace. Let `V_(Lambda,r)`
be the row basis returned by subduction for irrep `Lambda` and zero-based row
`r`. The kernel orthonormalizes the range of

```text
P_active @ V_(Lambda,r)
```

to obtain `W`, then constructs the selected quantization matrix as
`W† (R - B) W`. This is the intersection of the selected total-\( J \)
space with the requested irrep-row space. The box can still mix all active
sectors allowed by the irrep. If the requested irrep is absent from the
declared basis, or the selection removes its row entirely, evaluation fails
with a `ValueError`.

Nonzero amplitude mixing must preserve physical total parity,
`channel_intrinsic_parity[channel] * (-1)**ell`. The default common intrinsic
parity therefore forbids mixing opposite orbital parities; explicit channel
parities can allow such mixing when the total parities agree.

## Scope

This API implements a mathematically exact projection for the active space
specified by the caller. It does not decide whether a paper's omitted
partial waves or `J` blocks are physically negligible, map literature irrep
labels to package irreps, derive the reduced-inverse convention, normalize
the amplitude, or convert units. Those inputs and the physical case for the
truncation must be established by the calling analysis. Passing software
projection checks alone is not a paper reproduction or an independent
physics validation.
