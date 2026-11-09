"""Geometry-only SLD using the official PyTornado native vortex-lattice solver.

The geometry adapter below supplies a camber-surface lattice to PyTornado's
py2c_lattice/downwash/boundary/results routines and the same LAPACK solve used
by its Python driver. It does not implement an alternative induced-flow solver.
"""
from __future__ import annotations

import ctypes
from importlib.metadata import version
import math
from numbers import Integral
from types import SimpleNamespace

import numpy as np

import main as m


def _box(values):
    # PyTornado's native API accesses FixedOrderedDict._dictionary directly.
    return SimpleNamespace(_dictionary=values)


def _camber_surface(vertices, chord_panels, span_panels):
    xi = .5 * (1 - np.cos(np.linspace(0, np.pi, chord_panels + 1)))
    span_z = np.asarray(vertices[2, :, 0], dtype=float)
    m.require(np.allclose(vertices[2], span_z[:, None], rtol=0, atol=1e-8),
              "VLM adapter requires reference sections at constant span coordinates")
    m.require(bool((np.diff(span_z) > 0).all()), "VLM span stations must increase")
    sections = []
    for j in range(129):
        points = np.array(vertices[:2, j, :-1].T, dtype=float, copy=True)
        # SuperWing may split the trailing-edge closure with a midpoint. It is
        # not an extra upper-surface point when locating the mean trailing edge.
        if np.allclose(points[-1], .5 * (points[0] + points[-2]), rtol=0, atol=1e-8):
            points = points[:-1]
        te = .5 * (points[0] + points[-1])
        leading = int(np.argmin(points[:, 0]))
        axis = te - points[leading]
        leading = int(np.argmin(points @ axis))
        m.require(0 < leading < len(points)-1, "VLM leading edge must separate lower/upper surfaces")
        le = points[leading]
        chord = float(np.linalg.norm(te-le))
        m.require(chord > 0, "VLM section has zero chord")
        ex = (te-le)/chord
        ey = np.array([-ex[1], ex[0]])
        local_x = (points-le) @ ex/chord
        local_y = (points-le) @ ey/chord
        lower = m._profile_interp(local_x[:leading+1], local_y[:leading+1], xi)
        upper = m._profile_interp(local_x[leading:], local_y[leading:], xi)
        camber = le + chord*(xi[:, None]*ex + .5*(lower+upper)[:, None]*ey)
        # Original [x,height,span] -> PyTornado [x,span,height]. Mirror about
        # the actual exposed-root plane, consistent with the single half-wing.
        sections.append(np.stack((camber[:, 0], np.full(len(xi), span_z[j]-span_z[0]),
                                  camber[:, 1]), axis=-1))
    sections = np.asarray(sections)
    edges = np.linspace(span_z[0], span_z[-1], span_panels+1)
    mesh = np.empty((span_panels+1, chord_panels+1, 3), dtype=float)
    for i in range(chord_panels+1):
        for c in range(3):
            mesh[:, i, c] = np.interp(edges, span_z, sections[:, i, c])
    quads = np.stack((mesh[:-1, :-1], mesh[1:, :-1], mesh[1:, 1:], mesh[:-1, 1:]), axis=2)
    right = quads.reshape(-1, 4, 3)
    left = right[:, [1, 0, 3, 2]].copy()
    left[:, :, 1] *= -1
    points = np.ascontiguousarray(np.concatenate((right, left)), dtype=np.float64)
    chord_at_edges = np.linalg.norm(mesh[:, -1]-mesh[:, 0], axis=1)
    strip_chord = .5*(chord_at_edges[:-1]+chord_at_edges[1:])
    strip_width = np.diff(edges)
    return points, edges, strip_chord, strip_width, float(span_z[0])


def _solve_native(points, aoa_deg, half_area, half_span):
    try:
        from pytornado.aero import c_vlm
        from scipy.linalg.lapack import dgesv
        installed_version = version("pytornado")
    except (ImportError, OSError) as error:
        raise RuntimeError("Official PyTornado native solver is required. Use the flogen Conda "
                           "environment with the installed PyTornado build; no CFD-integral fallback is used.") from error
    n = len(points)
    info = {"num_wings": 1, "num_segments": n, "num_controls": 0,
            "num_panels": n, "num_strips": n,
            **{key: 0. for key in ("aspect_min", "aspect_max", "aspect_avg",
                                  "area_min", "area_max", "area_avg")}}
    lattice = SimpleNamespace(p=np.zeros((n, 4, 3)), v=np.zeros((n, 4, 3)),
                              c=np.zeros((n, 3)), bound_leg_midpoints=np.zeros((n, 3)),
                              n=np.zeros((n, 3)), a=np.zeros(n), epsilon=1e-6, info=_box(info))
    state = SimpleNamespace(
        aero=_box({"airspeed": 1., "density": 1., "alpha": float(aoa_deg), "beta": 0.,
                   "rate_P": 0., "rate_Q": 0., "rate_R": 0.}),
        refs=_box({"area": float(2*half_area), "span": float(2*half_span), "chord": 1.,
                   "gcenter": np.zeros(3), "rcenter": np.zeros(3)}))
    # Native PyTornado expects C long*, which is 32-bit on Windows, irrespective
    # of NumPy 2's default int64. Do not pass platform-default NumPy integers.
    long_dtype = np.dtype(ctypes.c_long)
    c_vlm.py2c_lattice(lattice, state, points, np.zeros(n, dtype=long_dtype),
                      np.ones((n, 2), dtype=long_dtype))
    m.require(bool(np.isfinite(lattice.a).all() and (lattice.a > 0).all()),
              "PyTornado produced invalid panel areas")
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)
    c_vlm.py2c_downwash(lattice, matrix)
    c_vlm.py2c_boundary(lattice, state, rhs)
    m.require(bool(np.isfinite(matrix).all() and np.isfinite(rhs).all()),
              "PyTornado produced non-finite influence/boundary values")
    _, _, circulation, status = dgesv(matrix, rhs)
    m.require(status == 0, f"PyTornado circulation solve failed: LAPACK info={status}")
    panelwise = {key: np.zeros(n) for key in
                 ("vx", "vy", "vz", "vmag", "fx", "fy", "fz", "fmag", "cp")}
    panelwise["gamma"] = np.ascontiguousarray(circulation)
    keys = ("x", "y", "z", "D", "C", "L", "l", "m", "n")
    result = SimpleNamespace(panelwise=_box(panelwise), forces=_box(dict.fromkeys(keys, 0.)),
                             coeffs=_box(dict.fromkeys(keys, 0.)))
    c_vlm.py2c_results(lattice, state, result)
    angle = math.radians(float(aoa_deg))
    lift = -panelwise["fx"]*math.sin(angle)+panelwise["fz"]*math.cos(angle)
    m.require(bool(np.isfinite(lift).all()), "Non-finite PyTornado panel lift")
    return lift, installed_version


def estimate_spanwise_lift(centers, vertices, mach, aoa_deg, chord_panels=4, span_panels=32):
    """Return (sectional_CL[1,128], provenance) without accepting a CFD field.

    The paper uses incompressible PyTornado VLM as a low-fidelity estimate. No
    unreported transonic/PG correction is added; Mach is recorded, not used to
    pretend the potential-flow estimate resolves shocks or viscous separation.
    """
    m.require(isinstance(chord_panels, Integral) and not isinstance(chord_panels, bool) and
              1 <= chord_panels <= 16, "chord_panels must be an integer in [1,16]")
    m.require(isinstance(span_panels, Integral) and not isinstance(span_panels, bool) and
              4 <= span_panels <= 128, "span_panels must be an integer in [4,128]")
    m.require(math.isfinite(float(mach)) and 0 <= float(mach) < 1,
              "VLM estimate expects a finite subsonic Mach")
    m.require(math.isfinite(float(aoa_deg)), "VLM angle of attack must be finite")
    centers, vertices = np.asarray(centers), np.asarray(vertices)
    features = m.geometry_features(centers, vertices)
    points, edges, chords, widths, root_z = _camber_surface(vertices, chord_panels, span_panels)
    half_area = float(np.sum(chords*widths))
    lift, installed_version = _solve_native(points, aoa_deg, half_area, float(edges[-1]-edges[0]))
    half_panels = span_panels*chord_panels
    right_lift = lift[:half_panels].reshape(span_panels, chord_panels).sum(axis=1)
    sectional_cl = right_lift/(.5*chords*widths)  # normalized rho=V=1, q=0.5
    stations = .5*(edges[:-1]+edges[1:])
    target = .5*(vertices[2, :-1, 0]+vertices[2, 1:, 0])
    # Root symmetry gives a finite root value; circulation falls to zero at the
    # free tip. Interpolate at physical span coordinates, never by array index.
    mapped = np.interp(target, np.r_[edges[0], stations, edges[-1]],
                       np.r_[sectional_cl[0], sectional_cl, 0.])[None, :]
    m.finite(mapped, "PyTornado sectional CL")
    total_cl = float(np.sum(right_lift)/(.5*half_area))
    return np.ascontiguousarray(mapped, dtype=np.float64), {
        "solver": "pytornado", "version": installed_version, "backend": "official_native_c_vlm",
        "method": "incompressible_horseshoe_vortex_lattice", "uses_cfd": False,
        "chord_panels": int(chord_panels), "span_panels_per_half": int(span_panels),
        "total_panels": len(points), "total_cl": total_cl, "mach": float(mach),
        "aoa_deg": float(aoa_deg), "compressibility_correction": "none",
        "camber_surface": "mean_of_lower_and_upper_at_equal_local_x_over_c",
        "symmetry_plane_original_z": root_z, "symmetry": "exposed_root_plane",
        "coordinate_mapping": "original_x_yheight_zspan_to_vlm_x_yspan_zheight",
        "output_quantity": "dimensionless_sectional_cl_before_sweep_transform",
        "output_shape": [1, 128], "interpolation": "physical_span_linear_root_symmetric_tip_zero",
        "reference_half_area_from_camber_chords": half_area,
        "limitations": "potential-flow thin-wing estimate; no shocks, separation or viscous effects",
    }
