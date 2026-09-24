"""Scoped left-hand-cut diagnostics; no exchange amplitude or modified QC.

For an elastic equal-mass channel with an allowed single t-channel exchange,
the endpoint t=-4k^2=mu^2 gives k_L^2=-mu^2/4, s_L=4m^2-mu^2.
Reference: arXiv:2311.18793v2, Eq. (2) and Sec. 2.2.
The cut excludes the standard QC at/below s_L; proximity from above can also
enhance neglected exponential corrections. This diagnostic does not bound them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite


class LeftHandCutDomainError(ValueError):
    """The requested method is excluded or cannot be assessed."""


@dataclass(frozen=True)
class LeftHandCutReport:
    qc_status: str
    ere_status: str
    path_status: str
    branch_point_s: float | None
    branch_point_k2: float | None
    minimum_s_margin: float | None
    maximum_ere_radius_ratio: float | None
    affected_s_interval: tuple[float, float] | None
    units: str
    reason: str
    source: str = "https://arxiv.org/html/2311.18793v2#S2.SS2"
    caveat: str = (
        "Only the declared exchange is checked. Other cuts, inelastic thresholds, "
        "ERE truncation error and enhanced finite-volume exponentials are not assessed."
    )

    def require_qc(self):
        if self.qc_status != "not_excluded":
            raise LeftHandCutDomainError(f"standard QC {self.qc_status}: {self.reason}")
        return self

    def require_ere(self):
        if self.ere_status not in ("inside_exchange_radius", "exchange_absent"):
            raise LeftHandCutDomainError(f"ERE {self.ere_status}: {self.reason}")
        return self

    def require_path(self):
        if self.path_status != "no_declared_cut_crossing":
            raise LeftHandCutDomainError(f"continuation path {self.path_status}: {self.reason}")
        return self


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return float(value)


def _crosses_cut(a, b, endpoint):
    """Intersect an entire straight segment, not just sampled path vertices."""
    if a.imag == b.imag == 0:
        return min(a.real, b.real) <= endpoint
    if a.imag == 0 and a.real <= endpoint or b.imag == 0 and b.real <= endpoint:
        return True
    if a.imag * b.imag < 0:
        fraction = -a.imag / (b.imag - a.imag)
        return (a + fraction * (b - a)).real <= endpoint
    return False


@dataclass(frozen=True)
class EqualMassExchangeDomain:
    """Single equal-mass elastic channel and declared t-channel exchange.

    ``exchange_allowed`` states the channel's selection rule; ``coupling`` is
    the real exchange coupling (only zero/nonzero matters). None means unknown.
    ``mechanism`` must be ``equal_mass_t_exchange`` for this analytic formula.
    Unknown mechanisms are retained as unknown, not assigned this branch point.
    Masses use the same ``units``; s windows/paths use their squared units.
    Sheets I/II label the two-body right-hand cut, NOT left-hand-cut windings.
    """
    mass: float
    exchange_mass: float | None = None
    exchange_allowed: bool | None = None
    coupling: float | None = None
    mechanism: str | None = None
    units: str = "temporal_lattice"

    def __post_init__(self):
        _positive(self.mass, "mass")
        if self.exchange_mass is not None:
            _positive(self.exchange_mass, "exchange_mass")
        if self.units not in ("temporal_lattice", "GeV"):
            raise ValueError("units must be temporal_lattice or GeV")
        if self.exchange_allowed is not None and type(self.exchange_allowed) is not bool:
            raise ValueError("exchange_allowed must be bool or None")
        if self.coupling is not None:
            if isinstance(self.coupling, bool) or not isinstance(self.coupling, (int, float)) or not isfinite(self.coupling):
                raise ValueError("coupling must be finite real or None")

    def to_temporal_lattice(self, at_gev_inverse):
        """Convert GeV masses using a_t in GeV^-1; convert s separately by a_t^2."""
        if self.units != "GeV":
            raise ValueError("conversion requires GeV input")
        scale = _positive(at_gev_inverse, "at_gev_inverse")
        return replace(self, mass=self.mass * scale,
                       exchange_mass=None if self.exchange_mass is None else self.exchange_mass * scale,
                       units="temporal_lattice")

    def check(self, s_window, *, path_s=None, sheet="I"):
        if len(s_window) != 2:
            raise ValueError("s_window must contain two endpoints")
        lo, hi = (_positive(x, "s endpoint") for x in s_window)
        if lo > hi:
            raise ValueError("s_window must be increasing (a single point is allowed)")
        path = None
        if path_s is not None:
            path = tuple(complex(x) for x in path_s)
            if not path or any(not isfinite(x.real) or not isfinite(x.imag) for x in path):
                raise ValueError("path_s must be a nonempty finite piecewise-linear path")

        def report(qc, ere, reason, *, point=None, k2=None, margin=None, ratio=None, affected=None, absent=False):
            path_status = "not_requested" if path is None else "unknown"
            if path is not None and sheet in ("I", "II") and (point is not None or absent):
                touches = not absent and (
                    any(x.imag == 0 and x.real <= point for x in path)
                    or any(_crosses_cut(a, b, point) for a, b in zip(path[:-1], path[1:]))
                )
                path_status = "crosses_declared_cut" if touches else "no_declared_cut_crossing"
            return LeftHandCutReport(qc, ere, path_status, point, k2, margin, ratio, affected, self.units, reason)

        if self.mechanism != "equal_mass_t_exchange":
            return report("unknown", "unknown", "missing or unsupported exchange mechanism")
        if self.exchange_allowed is False or self.coupling == 0:
            return report("not_excluded", "exchange_absent", "declared exchange is forbidden or decoupled", absent=True)
        if self.exchange_allowed is None or self.coupling is None or self.exchange_mass is None:
            return report("unknown", "unknown", "exchange mass, coupling and selection rule are required")
        mu = self.exchange_mass
        # Scope is the nearby single-exchange cut; do not extrapolate the
        # massive-identical-particle derivation through s=0 or unstable inputs.
        if mu >= 2*self.mass:
            return report("unknown", "unknown", "supported exchange range is 0 < mu < 2m")
        threshold = 4*self.mass**2
        branch = threshold - mu**2
        ratio = max(abs(lo-threshold), abs(hi-threshold)) / mu**2
        if path is not None:
            # |s-s_threshold| is convex along each straight segment, so its
            # maximum occurs at a vertex. ERE diagnostics include the path.
            ratio = max(ratio, max(abs(z-threshold) for z in path)/mu**2)
        return report(
            "excluded" if lo <= branch else "not_excluded",
            "outside_exchange_radius" if ratio >= 1 else "inside_exchange_radius",
            "equal-mass t exchange: s_L=4m^2-mu^2; k_L^2=-mu^2/4",
            point=branch, k2=-mu**2/4, margin=lo-branch, ratio=ratio,
            affected=(lo, min(hi, branch)) if lo <= branch else None,
        )
