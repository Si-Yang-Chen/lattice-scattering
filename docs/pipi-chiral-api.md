# Pi-pi chiral amplitude API

The product exposes the accepted two-flavor one-loop pi-pi amplitude through
`lattice_scattering.amplitudes`. The model is registered as
`pipi-nlo-chiral` and requires `pion_mass`,
`decay_constant`, `C1`, `C2`, and `C4`.

```python
from lattice_scattering.amplitudes import PipiNLOChiral, build

model = build("pipi-nlo-chiral", {
    "pion_mass": 0.14,
    "decay_constant": 0.092,
    "C1": 1.0, "C2": 1.0, "C4": 1.0,
})
model.inverse(4.0 * model.pion_mass**2)
```

All masses and momenta use one consistent natural-unit system. The fitted
direct observable is `k cot(delta) / pion_mass` as a function of physical
`s_phys`. Use `PipiChiralPhaseObservablePredictor` to preserve input row
order. For correlated fitting, construct an `ObservableDataset` with an
explicit full covariance and call `fit_pipi_chiral_observables`; pointwise
error constructors are rejected by that fit so covariance assumptions remain
visible.

The real-axis branch starts at `s_phys = 4*m_pi**2` and is valid below
`16*m_pi**2`. Subthreshold continuation is intentionally refused because no
continuation convention is selected by the product contract. Scan consumers
should split at `model.s_breakpoints()` and treat `2*m_pi**2` as an inverse
expansion pole.

The LEC helper `pipi_chiral_constants_from_ell_at_fpi` performs the scale-
`f_pi` conversion once. Callers that already have `C1`, `C2`, and `C4` should
pass those values directly.
