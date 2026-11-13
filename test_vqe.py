"""Grader-style tests for 03_vqe_molecular_energy.ipynb.

Usage (from the qiskit-quantum-hackathon folder, environment activated):
    python test_vqe.py                 # T1-T5 + one solve_geometry run (~5 min)
    python test_vqe.py --full          # + 4 solve_geometry runs like the grader (~10-20 min)
    python test_vqe.py --nb path/to/notebook.ipynb

The script loads the Provided and Participant cells of the notebook (it skips
the experiment cell VQE-X1), never modifies the notebook, and checks every task
with independent physical tests (HF energy, exact reference energy, spectra,
budget ledger, exact ideal/noisy energies of your returned state). It ends with
an estimated automatic score using the Section 11 formulas.
Keep this file OUT of your submission folder.
"""

import argparse
import contextlib
import io
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--nb", default="notebooks/03_vqe_molecular_energy.ipynb")
ap.add_argument("--full", action="store_true", help="4 solve_geometry runs (2 geometries x 2 seeds)")
ARGS = ap.parse_args()
NB = Path(ARGS.nb)


def load_namespace(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {}
    old = os.getcwd()
    os.chdir(nb_path.parent)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            first = src.strip().splitlines()[0] if src.strip() else ""
            if first.startswith("# Participant Code: Task VQE-X"):
                continue
            if first.startswith("# Provided Code") or first.startswith("# Participant Code"):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(src, f"<{first}>", "exec"), ns)
    finally:
        os.chdir(old)
    return ns


ns = load_namespace(NB)
g = ns.get
ph = ns["ph"]
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

DATA = {round(e["bond_length_angstrom"], 3): e for e in ph.load_public_integrals()}
HIDDEN_LIKE = [1.4, 2.6]  # used as stand-ins for the undisclosed geometries


def hidden_copy(entry):
    """Integrals as the grader passes them: no reference energies."""
    return {k: v for k, v in entry.items() if k not in ("e_hf", "e_ref")}


# ------------------------------------------------------------- exact helpers
def matrix(op):
    return op.to_matrix(sparse=False)


def number_sector_min(H6, n_orb=3, na=1, nb=1):
    """Lowest eigenvalue of the 6-qubit H restricted to na alpha, nb beta electrons (blocked order)."""
    M = matrix(H6)
    idx = [b for b in range(2 ** (2 * n_orb))
           if bin(b & ((1 << n_orb) - 1)).count("1") == na and bin(b >> n_orb).count("1") == nb]
    sub = M[np.ix_(idx, idx)]
    return float(np.linalg.eigvalsh(sub)[0])


def hf_energy(H6, n_orb=3):
    index = (1 << 0) | (1 << n_orb)  # qubits 0 and 3
    return float(np.real(matrix(H6)[index, index]))


def ideal_energy(circ, params, H):
    sv = Statevector(circ.assign_parameters(dict(zip(circ.parameters, params))))
    return float(np.real(sv.expectation_value(H)))


def noisy_energy(circ, params, H):
    rho = ph.noisy_density_matrix(circ, params)
    return float(np.real(np.trace(rho @ matrix(H))))


# ------------------------------------------------------------- runner
RESULTS = []


def test(name, points):
    def deco(fn):
        RESULTS.append((name, points, fn))
        return fn
    return deco


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as e:
        raise AssertionError(f"raised {type(e).__name__}, expected ValueError")
    raise AssertionError("did not raise ValueError")


SCORE = {}  # partial scores set by tests


# ------------------------------------------------------------- T1
@test("T1 Hamiltonian: HF energy, E_ref, hermitian, real coeffs (all public R)", 15)
def t1():
    build = g("build_qubit_hamiltonian")
    for R, e in DATA.items():
        H = build(hidden_copy(e))
        assert isinstance(H, SparsePauliOp), "return a SparsePauliOp"
        assert H.num_qubits == 6, f"R={R}: {H.num_qubits} qubits, expected 6"
        assert np.allclose(np.imag(H.coeffs), 0, atol=1e-12), f"R={R}: coefficients have imaginary parts"
        assert len(H) == len(H.simplify()), f"R={R}: not simplified (duplicate/zero terms)"
        ehf = hf_energy(H)
        assert abs(ehf - e["e_hf"]) < 1e-8, (
            f"R={R}: <HF|H|HF> on qubits 0,3 = {ehf:.10f}, expected e_hf {e['e_hf']:.10f} "
            f"(check blocked ordering, E_core, the 1/2 and (pq|rs) index pairing)")
        emin = number_sector_min(H)
        assert abs(emin - e["e_ref"]) < 1e-8, f"R={R}: sector minimum {emin:.10f} != e_ref {e['e_ref']:.10f}"
        gmin = float(np.linalg.eigvalsh(matrix(H))[0])
        assert abs(gmin - e["e_ref"]) < 1e-6, f"R={R}: global minimum {gmin:.8f} != e_ref (spectrum wrong)"


@test("T1 Hamiltonian: accepts nested lists; rejects bad shapes", 0)
def t1_edge():
    build = g("build_qubit_hamiltonian")
    e = DATA[1.6]
    H1 = build({"h1": e["h1"].tolist(), "eri": e["eri"].tolist(), "e_core": e["e_core"]})
    H2 = build(hidden_copy(e))
    assert np.allclose(matrix(H1), matrix(H2)), "nested lists give a different operator"
    expect_value_error(build, {"h1": np.zeros((3, 3)), "eri": np.zeros((2, 2, 2, 2)), "e_core": 0.0})
    expect_value_error(build, {"h1": np.zeros((3, 2)), "eri": np.zeros((3, 3, 3, 3)), "e_core": 0.0})


# ------------------------------------------------------------- T2
@test("T2 tapering: <6 qubits, eigenvalues subset of full, min = E_ref", 10)
def t2():
    build, taper = g("build_qubit_hamiltonian"), g("taper_hamiltonian")
    for R in (1.0, 1.6, 3.0) + tuple(HIDDEN_LIKE):
        e = DATA[R]
        H = build(hidden_copy(e))
        T = taper(H, 1, 1)
        assert isinstance(T, SparsePauliOp) and T.num_qubits < 6, f"R={R}: tapered op has {T.num_qubits} qubits"
        full = np.linalg.eigvalsh(matrix(H))
        tap = np.linalg.eigvalsh(matrix(T))
        for v in tap:
            assert np.min(np.abs(full - v)) < 1e-6, f"R={R}: tapered eigenvalue {v:.6f} not in full spectrum"
        assert abs(tap[0] - e["e_ref"]) < 1e-6, f"R={R}: tapered min {tap[0]:.8f} != e_ref {e['e_ref']:.8f} (wrong sector?)"
    print(f"      tapered: {T.num_qubits} qubits, {len(T)} terms")


# ------------------------------------------------------------- T3
def check_groups(H, groups):
    n = H.num_qubits
    total = None
    seen = 0
    for basis, grp in groups:
        assert isinstance(basis, str) and len(basis) == n and set(basis) <= set("XYZ"), f"bad basis {basis!r}"
        for label in grp.paulis.to_labels():
            for q, (pl, bl) in enumerate(zip(label, basis)):
                assert pl == "I" or pl == bl, f"term {label} incompatible with basis {basis}"
        seen += len(grp)
        total = grp if total is None else total + grp
    assert seen == len(H.simplify()), f"groups contain {seen} terms, H has {len(H.simplify())} (duplicates or missing)"
    assert np.allclose(matrix(total.simplify()), matrix(H)), "groups do not sum to H"
    return len(groups)


@test("T3 grouping: valid and complete on 6- and 4-qubit Hamiltonians", 8)
def t3():
    build, group = g("build_qubit_hamiltonian"), g("group_qubitwise_commuting")
    rng = np.random.default_rng(3)
    ops = []
    for R in (1.0, 2.6):
        H = build(hidden_copy(DATA[R]))
        ops.append(H)
        try:
            ops.append(g("taper_hamiltonian")(H, 1, 1))
        except Exception:
            pass
    for _ in range(3):  # random operators
        labels = list({"".join(rng.choice(list("IXYZ"), 4)) for _ in range(25)})
        ops.append(SparsePauliOp(labels, rng.normal(size=len(labels))))
    sizes = [check_groups(H, group(H)) for H in ops]
    print(f"      number of groups: {sizes} (fewer = more shots per group)")


# ------------------------------------------------------------- T4
@test("T4 expectation_from_counts: exact on synthetic counts + edge cases", 4)
def t4a():
    f = g("expectation_from_counts")
    # single Z on qubit 0, coefficient 2: v = +2 for b0=0, -2 for b0=1
    m, se = f(SparsePauliOp(["IZ"], [2.0]), {"00": 30, "01": 10})
    exp_mean = (30 * 2 + 10 * -2) / 40
    var = (30 * (2 - exp_mean) ** 2 + 10 * (-2 - exp_mean) ** 2) / 39  # sample variance (ddof=1)
    var0 = (30 * (2 - exp_mean) ** 2 + 10 * (-2 - exp_mean) ** 2) / 40  # population (ddof=0)
    assert abs(m - exp_mean) < 1e-9, f"mean {m}, expected {exp_mean} (bit order: last char = qubit 0)"
    # the notebook's public check uses the population variance (divide by m, ddof=0)
    assert abs(se - math.sqrt(var0 / 40)) < 1e-9, (
        f"SE {se}, expected {math.sqrt(var0/40):.6f} (population variance / m; "
        f"{math.sqrt(var/40):.6f} would be ddof=1, which the public check rejects)")
    # ZZ + identity: v = 1.5 + 0.5*(-1)^(b0+b1)
    m, _ = f(SparsePauliOp(["II", "ZZ"], [1.5, 0.5]), {"00": 1, "11": 1, "01": 2})
    assert abs(m - (2.0 + 2.0 + 2 * 1.0) / 4) < 1e-9, f"identity/ZZ mean wrong: {m}"
    # X on qubit 1 measured in basis 'XZ' -> parity of bit for qubit 1 (first character)
    m, _ = f(SparsePauliOp(["XI"], [1.0]), {"10": 3, "00": 1})
    assert abs(m - (-3 + 1) / 4) < 1e-9, f"qubit-1 parity wrong: {m}"
    for bad in ({}, {"000": 5}):
        expect_value_error(f, SparsePauliOp(["IZ"], [1.0]), bad)


@test("T4 estimate_energy: one job, unbiased within 5 sigma", 3)
def t4b():
    est, build = g("estimate_energy"), g("build_qubit_hamiltonian")
    H = build(hidden_copy(DATA[1.6]))
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    a, b = Parameter("a"), Parameter("b")
    c = QuantumCircuit(6)
    c.x(0); c.x(3); c.ry(a, 1); c.cx(1, 4); c.ry(b, 2)
    params = [0.4, -0.3]
    s = ph.BudgetedSampler(seed=5)
    E, se = est(H, c, params, s, 20000)
    assert s.jobs == 1, f"estimate_energy used {s.jobs} sampler jobs, must be exactly 1"
    exact = noisy_energy(c, params, H)
    assert 0 < se < 0.05, f"standard error {se} looks wrong"
    assert abs(E - exact) < 5 * se + 1e-6, f"estimate {E:.5f} vs exact noisy {exact:.5f} ({abs(E-exact)/se:.1f} sigma)"


# ------------------------------------------------------------- T5 (analysis, exact tools allowed here)
@test("T5 ansatz: validity, <=40 CX, can reach E_ref exactly (statevector optimum)", 0)
def t5():
    build_ansatz, build = g("build_ansatz"), g("build_qubit_hamiltonian")
    for nq in (6, 4):
        try:
            c = build_ansatz(nq)
        except Exception as ex:
            print(f"      build_ansatz({nq}) raised {type(ex).__name__}: {ex}")
            continue
        assert c.num_qubits == nq and c.num_clbits == 0, f"{nq}: wrong registers"
        assert not {"measure", "reset"} & set(c.count_ops()), "contains measure/reset"
        cx = ph.count_two_qubit_gates(c)
        assert cx <= 40, f"{nq} qubits: {cx} two-qubit gates > 40"
        print(f"      {nq} qubits: {c.num_parameters} parameters, {cx} two-qubit gates")
    # exact optimum for the 6-qubit Hamiltonian (or tapered, if that is what solve_geometry uses)
    for R in (1.0, 1.6, 3.0):
        e = DATA[R]
        H = build(hidden_copy(e))
        c = build_ansatz(6)
        z = np.zeros(c.num_parameters)
        e0 = ideal_energy(c, z, H)
        best = math.inf
        rng = np.random.default_rng(0)
        for start in [z] + [rng.normal(scale=0.3, size=z.size) for _ in range(4)]:
            r = minimize(lambda p: ideal_energy(c, p, H), start, method="BFGS")
            best = min(best, r.fun)
        err = 1000 * (best - e["e_ref"])
        print(f"      R={R}: E(theta=0)-E_HF = {1000*(e0-e['e_hf']):+.3f} mHa, "
              f"best ideal error {err:.4f} mHa (target < 1.6), noisy shift at optimum not shown")
        assert err < 1.6, f"R={R}: ansatz cannot reach chemical accuracy even noiselessly ({err:.3f} mHa)"


# ------------------------------------------------------------- T6 + scoring
def score_run(entry, out, sampler, wall):
    """Section 11 formulas for one run. Returns dict of partial scores."""
    s = {"C5": 0.0, "C6": 0.0, "C7": 0.0, "C8": 0.0, "C9": 0.0}
    need = {"ansatz", "parameters", "energy", "std_error", "hamiltonian", "history"}
    missing = need - set(out)
    assert not missing, f"missing keys {missing}"
    c, p, H = out["ansatz"], np.asarray(out["parameters"], float), out["hamiltonian"]
    # Hamiltonian validity
    full = np.linalg.eigvalsh(matrix(g("build_qubit_hamiltonian")(hidden_copy(entry))))
    ev = np.linalg.eigvalsh(matrix(H))
    valid_h = all(np.min(np.abs(full - v)) < 1e-6 for v in ev) and abs(ev[0] - entry["e_ref"]) < 1e-6
    # C5
    shape_ok = c.num_qubits == H.num_qubits and c.num_parameters == p.size and c.num_clbits == 0
    cx = ph.count_two_qubit_gates(c)
    s["C5"] = 0.4 * shape_ok + 0.6 * (cx <= 40)
    # C6
    evaluated = any(np.allclose(le.parameters, p, atol=1e-12) for le in sampler.ledger)
    within = (sampler.jobs <= 600 and sampler.circuit_executions <= 20000
              and sampler.total_shots <= 20_000_000)
    s["C6"] = 0.4 * (wall <= 300) + 0.4 * evaluated + 0.2 * within
    if not (valid_h and cx <= 40 and wall <= 300):
        return s, dict(valid_h=valid_h, cx=cx, wall=wall)
    ehf = entry["e_hf"]
    d_hf = 1000 * (ehf - entry["e_ref"])
    d_id = 1000 * (ideal_energy(c, p, H) - entry["e_ref"])
    d_no = 1000 * (noisy_energy(c, p, H) - entry["e_ref"])
    l = math.log10
    clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
    if d_id < -1e-3:
        info = "IDEAL ENERGY BELOW E_REF -> flagged, C7/C8 = 0"
    else:
        s["C7"] = clip((l(d_hf) - l(max(d_id, 1.6))) / (l(d_hf) - l(1.6)))
        s["C8"] = clip((l(d_hf) - l(max(d_no, 5.0))) / (l(d_hf) - l(5.0)))
        info = ""
    E_noisy = entry["e_ref"] + d_no / 1000
    se = float(out["std_error"])
    s["C9"] = float(se > 0 and math.isfinite(se) and abs(out["energy"] - E_noisy) <= max(4 * se, 1e-3))
    return s, dict(valid_h=valid_h, cx=cx, wall=wall, d_ideal_mHa=d_id, d_noisy_mHa=d_no,
                   d_hf_mHa=d_hf, energy=out["energy"], std_error=se, evaluated=evaluated,
                   usage=sampler.usage, info=info)


@test("T6 solve_geometry: grader-style runs and Section 11 score", 0)
def t6():
    solve = g("solve_geometry")
    runs = [(R, sd) for R in HIDDEN_LIKE for sd in (101, 202)] if ARGS.full else [(2.6, 101)]
    totals = {k: [] for k in ("C5", "C6", "C7", "C8", "C9")}
    for R, sd in runs:
        entry = DATA[R]
        sampler = ph.BudgetedSampler(seed=sd)
        t0 = time.monotonic()
        try:
            out = solve(hidden_copy(entry), sampler, sd)
            wall = time.monotonic() - t0
            s, info = score_run(entry, out, sampler, wall)
        except Exception as ex:
            wall = time.monotonic() - t0
            print(f"      R={R} seed={sd}: {type(ex).__name__}: {ex} after {wall:.0f}s -> run scores 0")
            s = {k: 0.0 for k in totals}
            info = {}
        for k in totals:
            totals[k].append(s[k])
        if info:
            print(f"      R={R} seed={sd}: {wall:.0f}s, CX={info['cx']}, usage={info.get('usage')}")
            if "d_ideal_mHa" in info:
                print(f"         dE_ideal={info['d_ideal_mHa']:.3f} mHa (target<=1.6)  "
                      f"dE_noisy={info['d_noisy_mHa']:.3f} mHa (target<=5)  dE_HF={info['d_hf_mHa']:.2f} mHa")
                print(f"         E_hat={info['energy']:.6f} +- {info['std_error']*1000:.3f} mHa, "
                      f"final params evaluated: {info['evaluated']} {info['info']}")
    weights = {"C5": 10, "C6": 10, "C7": 15, "C8": 10, "C9": 5}
    for k, w in weights.items():
        SCORE[k] = w * float(np.mean(totals[k]))
    print("      " + "  ".join(f"{k}={SCORE[k]:.1f}/{w}" for k, w in weights.items()))


# ------------------------------------------------------------- main
if __name__ == "__main__":
    auto = {"T1 Hamiltonian: HF": 15, "T2": 10, "T3": 8, "T4 expectation": 4, "T4 estimate": 3}
    earned = 0.0
    for name, pts, fn in RESULTS:
        t0 = time.time()
        try:
            fn()
            status = "PASS"
            earned += pts
        except NotImplementedError:
            status = "TODO"
        except AssertionError as e:
            status = f"FAIL: {e}"
        except Exception as e:
            status = f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc(limit=2)
        print(f"{status.split(':')[0]:<5} [{pts:>2}] {name} ({time.time()-t0:.0f}s)"
              + (f"\n      -> {status}" if status not in ("PASS", "TODO") else ""))
    total = earned + sum(SCORE.values())
    print(f"\nEstimated automatic score: {total:.1f}/90  "
          f"(T1-T4: {earned:.0f}/40, T5/T6 formulas: {sum(SCORE.values()):.1f}/50"
          f"{'' if ARGS.full else ', from 1 run; use --full for 4 runs'})")
    print("Note: 1.4 and 2.6 A are public stand-ins; the real hidden geometries differ.")
