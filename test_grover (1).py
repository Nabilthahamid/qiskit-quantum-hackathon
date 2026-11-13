"""Hidden-style tests for 02_grover_search_under_noise.ipynb.

Usage (from the qiskit-quantum-hackathon folder, environment activated):
    python test_grover.py                       # tests notebooks/02_grover_search_under_noise.ipynb
    python test_grover.py path/to/notebook.ipynb

The script loads the Provided and Participant cells of the notebook (it skips the
experiment cell GRV-X1) and runs tests that mimic the hidden grader:
random keys of 2-6 bits, edge cases, noise models and the shot budget.
It never modifies the notebook. Keep this file OUT of your submission folder.
"""

import contextlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

NB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/02_grover_search_under_noise.ipynb")
RNG = random.Random(12345)


# ---------------------------------------------------------------- loader
def load_namespace(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {}
    old_cwd = os.getcwd()
    os.chdir(nb_path.parent)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            first = src.strip().splitlines()[0] if src.strip() else ""
            if first.startswith("# Participant Code: Task GRV-X"):
                continue  # skip experiments
            if first.startswith("# Provided Code") or first.startswith("# Participant Code"):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(src, f"<{first}>", "exec"), ns)
    finally:
        os.chdir(old_cwd)
    return ns


# ---------------------------------------------------------------- tiny runner
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
    except Exception as e:  # wrong exception type
        raise AssertionError(f"{fn.__name__}{args!r} raised {type(e).__name__}, expected ValueError")
    raise AssertionError(f"{fn.__name__}{args!r} did not raise ValueError")


def random_key(n):
    return "".join(RNG.choice("01") for _ in range(n))


def equal_up_to_phase(u, v, tol=1e-8):
    idx = np.unravel_index(np.argmax(np.abs(v)), v.shape)
    if abs(u[idx]) < 1e-12:
        return False
    phase = u[idx] / v[idx]
    return abs(abs(phase) - 1) < tol and np.allclose(u, phase * v, atol=tol)


def no_measure_copy(circ):
    c = circ.remove_final_measurements(inplace=False)
    return c


ns = load_namespace(NB)
g = ns.get
from qiskit.quantum_info import Operator, Statevector, DensityMatrix  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402


# ---------------------------------------------------------------- T1
@test("T1 oracle: unitary for random keys n=2..6 (+ all keys n=2,3)", 15)
def t1_unitary():
    keys = [format(i, f"0{n}b") for n in (2, 3) for i in range(2 ** n)]
    keys += [random_key(n) for n in (4, 5, 6) for _ in range(4)]
    for key in keys:
        n = len(key)
        c = g("build_oracle")(key)
        assert c.num_qubits == n, f"key {key}: {c.num_qubits} qubits, expected {n}"
        assert c.num_clbits == 0, f"key {key}: oracle must have no classical bits"
        assert "measure" not in c.count_ops(), f"key {key}: oracle contains measurements"
        ideal = np.eye(2 ** n, dtype=complex)
        ideal[int(key, 2), int(key, 2)] = -1  # Qiskit order: int(key, 2) is the basis index
        assert equal_up_to_phase(Operator(c).data, ideal), f"key {key}: wrong unitary (check bit order!)"


@test("T1 oracle: input validation", 0)
def t1_errors():
    f = g("build_oracle")
    for bad in (101, ["0", "1"], "1", "012", "01a", "", "0" * 9, None):
        expect_value_error(f, bad)


# ---------------------------------------------------------------- T2
@test("T2 diffuser: unitary n=2..8", 10)
def t2_unitary():
    for n in range(2, 9):
        c = g("build_diffuser")(n)
        assert c.num_qubits == n and c.num_clbits == 0, f"n={n}: wrong register sizes"
        s = np.full(2 ** n, 1 / math.sqrt(2 ** n))
        ideal = 2 * np.outer(s, s) - np.eye(2 ** n)
        assert equal_up_to_phase(Operator(c).data, ideal), f"n={n}: not 2|s><s| - I"


@test("T2 diffuser: input validation", 0)
def t2_errors():
    for bad in (1, 9, 0, -3, 2.5, "4"):
        expect_value_error(g("build_diffuser"), bad)


# ---------------------------------------------------------------- T3
@test("T3 grover circuit: exact P_r for random keys and r", 15)
def t3_prob():
    for _ in range(14):
        n = RNG.randint(2, 6)
        key = random_key(n)
        r = RNG.randint(0, 6)
        c = g("build_grover_circuit")(key, r)
        assert c.num_qubits == n and c.num_clbits == n, f"{key}, r={r}: wrong register sizes"
        probs = Statevector(no_measure_copy(c)).probabilities_dict()
        p = probs.get(key, 0.0)
        theta = math.asin(math.sqrt(1 / 2 ** n))
        ideal = math.sin((2 * r + 1) * theta) ** 2
        assert abs(p - ideal) < 1e-8, f"key {key}, r={r}: P={p:.6f}, expected {ideal:.6f}"


@test("T3 grover circuit: qubit i measured into clbit i", 0)
def t3_measure_map():
    key = "0010"
    c = g("build_grover_circuit")(key, 3)
    for inst in c.data:
        if inst.operation.name == "measure":
            q = c.find_bit(inst.qubits[0]).index
            b = c.find_bit(inst.clbits[0]).index
            assert q == b, f"qubit {q} measured into clbit {b}"
    counts = g("run_counts")(c, 2000, 1, None)
    top = max(counts, key=counts.get)
    assert top == key, f"most frequent outcome {top}, expected {key}"


@test("T3 grover circuit: input validation", 0)
def t3_errors():
    f = g("build_grover_circuit")
    for bad in (-1, 1.5, "2", None):
        expect_value_error(f, "0101", bad)


# ---------------------------------------------------------------- T4 / T5
@test("T4 ideal_success_probability and optimal_iterations n=2..10", 8)
def t4():
    for n in range(2, 11):
        theta = math.asin(math.sqrt(1 / 2 ** n))
        for r in range(0, 30):
            exp = math.sin((2 * r + 1) * theta) ** 2
            got = g("ideal_success_probability")(n, r)
            assert abs(got - exp) < 1e-9, f"P(n={n}, r={r}) = {got}, expected {exp}"
        # first peak only: r in 0 .. floor(pi/(4 theta)) (later periods are not "Grover's optimum")
        ps = [round(math.sin((2 * r + 1) * theta) ** 2, 12) for r in range(int(math.pi / (4 * theta)) + 1)]
        best = ps.index(max(ps))
        got = g("optimal_iterations")(n)
        assert got == best and isinstance(got, (int, np.integer)), f"optimal_iterations({n}) = {got}, expected {best}"


@test("T5 classical_success_probability", 4)
def t5():
    for n in range(2, 11):
        N = 2 ** n
        for k in (0, 1, 2, 5, N - 2, N - 1, N, N + 7):
            if k < 0:
                continue
            exp = 1.0 if k >= N - 1 else (k + 1) / N
            got = g("classical_success_probability")(n, k)
            assert abs(got - exp) < 1e-9, f"n={n}, k={k}: {got}, expected {exp}"
    expect_value_error(g("classical_success_probability"), 4, -1)


# ---------------------------------------------------------------- T6 / T7
def dm_after(circ, model):
    c = circ.copy()
    c.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=model)
    return np.asarray(sim.run(c, optimization_level=0).result().data()["density_matrix"])


def x_circ():
    c = QuantumCircuit(1); c.x(0); return c


def cx_circ():
    c = QuantumCircuit(2); c.cx(0, 1); return c


def check_model_common(build, name):
    for bad in (-0.01, 1.01):
        expect_value_error(build, bad)
    m = build(0.0)
    assert set(m.basis_gates) >= {"rz", "sx", "x", "cx"}, f"{name}: basis gates {m.basis_gates}"
    for circ in (x_circ(), cx_circ()):
        rho = dm_after(circ, m)
        ideal = np.asarray(DensityMatrix(circ).data)
        assert np.allclose(rho, ideal, atol=1e-10), f"{name}(0) changes the circuit"
    m = build(0.3)
    assert "rz" not in m.noise_instructions, f"{name}: rz must be error-free"


@test("T6 depolarizing model D(p)", 8)
def t6():
    build = g("build_depolarizing_noise_model")
    check_model_common(build, "D")
    for p in (0.05, 0.2, 0.7):
        m = build(p)
        rho = dm_after(x_circ(), m)
        lam = p / 10
        exp = (1 - lam) * np.diag([0, 1]) + lam * np.eye(2) / 2
        assert np.allclose(rho, exp, atol=1e-9), f"D({p}) on x is wrong (expected lambda=p/10)"
        rho = dm_after(cx_circ(), m)
        exp = (1 - p) * np.diag([1, 0, 0, 0]) + p * np.eye(4) / 4
        assert np.allclose(rho, exp, atol=1e-9), f"D({p}) on cx is wrong"
        sx = QuantumCircuit(1); sx.sx(0)
        ideal = np.asarray(DensityMatrix(sx).data)
        exp = (1 - lam) * ideal + lam * np.eye(2) / 2
        assert np.allclose(dm_after(sx, m), exp, atol=1e-9), f"D({p}) on sx is wrong"


@test("T7 bit-flip model B(q)", 8)
def t7():
    build = g("build_bitflip_noise_model")
    check_model_common(build, "B")
    for q in (0.05, 0.2, 0.5):
        m = build(q)
        d = np.real(np.diag(dm_after(x_circ(), m)))
        assert np.allclose(d, [q, 1 - q], atol=1e-9), f"B({q}) on x: populations {d}"
        d = np.real(np.diag(dm_after(cx_circ(), m)))
        exp = [(1 - q) ** 2, q * (1 - q), q * (1 - q), q * q]  # |00>,|01>,|10>,|11>
        assert np.allclose(d, exp, atol=1e-9), f"B({q}) on cx: populations {np.round(d,5)}, expected {np.round(exp,5)} (independent flips?)"


# ---------------------------------------------------------------- T8
@test("T8 success_probability on synthetic counts", 0)
def t8a():
    f = g("success_probability")
    assert abs(f({"0110": 30, "0000": 70}, "0110") - 0.3) < 1e-12
    assert f({"0000": 5}, "0110") == 0.0
    assert abs(f({"11": 1}, "11") - 1.0) < 1e-12
    expect_value_error(f, {}, "0110")


@test("T8 run_noisy_grover statistics (5 sigma)", 10)
def t8b():
    run = g("run_noisy_grover")
    ledger = g("SHOT_LEDGER")
    for key, r, model in (("101", 2, g("build_depolarizing_noise_model")(0.0)),
                          ("0110", 3, g("build_depolarizing_noise_model")(0.0)),
                          ("11010", 4, g("build_depolarizing_noise_model")(0.0))):
        shots = 4000
        ledger.reset()
        p = run(key, r, model, shots, 7)
        assert ledger.shots == shots, f"used {ledger.shots} shots via run_counts, expected {shots}"
        theta = math.asin(math.sqrt(1 / 2 ** len(key)))
        ideal = math.sin((2 * r + 1) * theta) ** 2
        se = math.sqrt(max(ideal * (1 - ideal), 1e-4) / shots)
        assert abs(p - ideal) < 5 * se, f"{key}, r={r}: {p:.4f} vs ideal {ideal:.4f}"
    # noisy: should drop toward 1/N and be reproducible with the same seed
    m = g("build_depolarizing_noise_model")(0.05)
    a = run("0110", 3, m, 2000, 11)
    b = run("0110", 3, m, 2000, 11)
    assert a == b, "same seed gave different results (pass seed to run_counts)"
    assert a < 0.9, f"strong noise should reduce success, got {a}"


# ---------------------------------------------------------------- T9
@test("T9 best_iteration_count: budget, format, accuracy", 12)
def t9():
    best = g("best_iteration_count")
    ledger = g("SHOT_LEDGER")
    cases = [("0110", g("build_depolarizing_noise_model")(0.0), 6000),
             ("101", g("build_depolarizing_noise_model")(0.02), 3000),
             ("1001", g("build_bitflip_noise_model")(0.01), 5000),
             ("11010", g("build_depolarizing_noise_model")(0.01), 8000),
             ("01", g("build_bitflip_noise_model")(0.0), 997)]
    for key, model, budget in cases:
        n = len(key)
        rstar = g("optimal_iterations")(n)
        ledger.reset()
        t0 = time.time()
        out = best(key, model, budget, 2026)
        dt = time.time() - t0
        assert isinstance(out, dict), "must return a dict"
        assert set(out) >= {"best_iterations", "success_probabilities", "shots_used"}, f"keys: {set(out)}"
        assert ledger.shots <= budget, f"{key}: ledger {ledger.shots} > budget {budget} (case scores 0!)"
        assert out["shots_used"] == ledger.shots, f"shots_used {out['shots_used']} != ledger {ledger.shots}"
        rbest = out["best_iterations"]
        assert isinstance(rbest, (int, np.integer)) and 0 <= rbest <= rstar + 2, f"best r={rbest} outside 0..{rstar+2}"
        # exact noisy probabilities of each candidate (density matrix)
        ofl = g("official_transpile")
        exact = {}
        for cand in range(rstar + 3):
            c = g("build_grover_circuit")(key, cand)
            c = ofl(c)
            sim = AerSimulator(method="density_matrix", noise_model=model)
            c2 = c.remove_final_measurements(inplace=False)
            c2.save_probabilities(label="pr")
            pr = np.asarray(sim.run(c2, optimization_level=0).result().data()["pr"])
            exact[cand] = float(pr[int(key, 2)])
        pmax = max(exact.values())
        se = math.sqrt(max(pmax * (1 - pmax), 1e-4) / (budget / (rstar + 3)))
        assert exact[rbest] >= pmax - 3 * math.sqrt(2) * se, (
            f"{key}: chose r={rbest} (P={exact[rbest]:.3f}), best is {max(exact, key=exact.get)} (P={pmax:.3f})")
        print(f"      {key}: r={rbest}, shots {ledger.shots}/{budget}, {dt:.1f}s")


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    earned = total = 0
    for name, pts, fn in RESULTS:
        total += pts
        try:
            fn()
            earned += pts
            print(f"PASS  [{pts:>2}] {name}")
        except NotImplementedError:
            print(f"TODO  [{pts:>2}] {name}: not implemented")
        except AssertionError as e:
            print(f"FAIL  [{pts:>2}] {name}: {e}")
        except Exception as e:
            print(f"ERROR [{pts:>2}] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\nEstimated automatic score: {earned}/{total} (validation tests are 0-pt but can cost hidden cases)")
