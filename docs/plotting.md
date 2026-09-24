# Plotting calculated results

Install the optional plotting dependency and generate the included synthetic
gallery from the current source checkout:

```sh
python -m pip install -e ".[plot]"
python -m lattice_scattering plot-demo examples/mock_plots
python -m lattice_scattering plot-results examples/mock_plots/mock-results.json examples/replotted
```

The command calculates a two-channel pole-plus-background toy model with the
package's Chew–Mandelstam phase space. It solves finite-volume levels in rest
and moving frames at several volumes, samples the real-axis amplitude, checks
the row-matrix eigenvalues and root residuals, and refines three local complex
poles after varying a toy coupling. A fixed random seed adds correlated noise
to the calculated levels. `mock-results.json` records the model, inputs,
calculated arrays, synthetic observations, covariance, and output filenames.
Masked values appear as JSON `null`. Each figure is written as PNG and SVG.
The second command renders this saved `lattice-scattering-plot-results/v1`
record without rerunning the scattering calculation. Other analyses can write
the same declared result schema, including `units`, `provenance`, channel masses,
calculated arrays and covariance, then use `plot-results` as a standalone
renderer. It checks shapes and finite inputs but does not certify the physical
model or infer omitted convention metadata.

The five images are:

| File | Content |
| --- | --- |
| `spectrum` | Synthetic observations and model roots against volume, with thresholds and **unprojected** non-interacting reference levels |
| `amplitudes` | Open-channel `rho_i rho_j |t_ij|²` in this model's declared amplitude convention; grey regions are closed |
| `root_diagnostic` | Scaled Hermitian eigenbranches, reported roots, nearby free references and model breakpoints, and each root's backward residual |
| `poles` | Locally refined poles on the declared `(II, II)` sheet for a coupling sweep, plus the absolute residue matrix of the reference model |
| `fit_diagnostics` | Cholesky-whitened differences between synthetic observations and model truth, and correlation-matrix eigenvalues; **no fit or eigenmode cutoff is performed** |

The checked-in gallery includes [spectrum](../examples/mock_plots/spectrum.png),
[amplitudes](../examples/mock_plots/amplitudes.png),
[root diagnostic](../examples/mock_plots/root_diagnostic.png),
[poles](../examples/mock_plots/poles.png), and
[residual/covariance diagnostic](../examples/mock_plots/fit_diagnostics.png).

All figures say **SYNTHETIC MOCK**. The layout is inspired by the spectrum,
amplitude and pole figures of
[Wilson et al., arXiv:2309.14071v1](https://arxiv.org/html/2309.14071v1),
but neither the parameters nor the synthetic data reproduce that paper.

For your own results, import the builders from `lattice_scattering.plotting`:
`spectrum_figure`, `amplitude_matrix_figure`, `root_diagnostic_figure`,
`pole_map_figure`, and `fit_diagnostic_figure`. Each requires an explicit
`provenance` label and returns a Matplotlib `Figure`; call `save_figure` to write
PNG, SVG, or PDF. The builders consume calculated arrays and do not infer
paper-specific K-matrix normalization, energy units, irrep mapping, sheet
continuation, or a root's physical identity.

The free-state enumerator does not project into an irrep or symmetrize
identical-particle exchanges, so its points are reference energies only.
Finite sampling does not certify all quantization roots. `solve_poles` locally
refines simple poles from supplied seeds; the toy coupling sweep is not a
statistical uncertainty band. The covariance plot shows eigenvalue guides but
does not apply the paper's eigenmode-cutoff fitting method.

The JSON record declares `schema`, `units`, `provenance`, channel labels and
masses, `spectrum_panels`, real-axis `amplitude`, `root_diagnostic`, `poles`,
ordered observations, predictions and covariance. The example record is the
concrete schema template. `null` denotes a deliberate display mask at a closed
channel or declared singular point. For a non-mock analysis, replace the arrays
with independently calculated results and set an accurate visible provenance
label; changing the Boolean `synthetic` field alone does not validate data.

The Python interface can also construct an individual figure:

```python
from lattice_scattering.plotting import SpectrumPanel, SpectrumPoint, save_figure, spectrum_figure

panel = SpectrumPanel(
    "rest A1g",
    observations=(SpectrumPoint(20, 0.721, 0.002),),
    predictions=(SpectrumPoint(20, 0.720),),
    free_references=(SpectrumPoint(20, 0.710, channel="A+A"),),
    thresholds_at={"A+A": 0.600},
)
figure = spectrum_figure([panel], provenance="SYNTHETIC MOCK · not paper data")
save_figure(figure, "figures/levels", formats=("png", "pdf"))
```
