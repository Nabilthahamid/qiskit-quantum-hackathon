"""Grader-style tests for 01_e91_quantum_key_distribution.ipynb.

Usage (from the qiskit-quantum-hackathon folder):
    python test_e91.py [path/to/notebook.ipynb]
Never modifies the notebook. Keep this file OUT of your submission folder.
"""

import contextlib
import io
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

NB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/01_e91_quantum_key_distribution.ipynb")


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
            if first.startswith("# Participant Code: Task E91-X"):
                continue
            if first.startswith("# Provided Code") or first.startswith("# Participant Code"):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(src, f"<{first}>", "exec"), ns)
    finally:
        os.chdir(old)
    return ns


ns = load_namespace(NB)
g = ns.get
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector, DensityMatrix  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

A, B, EVE = list(ns["ALICE_ANGLES"]), list(ns["BOB_ANGLES"]), list(ns["EVE_ANGLES"])
CHSH = (((2, 0), 1), ((2, 2), 1), ((0, 0), 1), ((0, 2), -1))
KEY_PAIRS = ((1, 0), (2, 1))
RNG = np.random.default_rng(7)
SIM = AerSimulator(seed_simulator=11)

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
        raise AssertionError(f"{fn.__name__}: raised {type(e).__name__}, expected ValueError")
    raise AssertionError(f"{fn.__name__}{tuple(type(a).__name__ for a in args)} did not raise ValueError")


def exact_E(a, b, p, lam):
    """Independent exact model of Section 5.6 (density matrices, no participant code)."""
    def ry(t):
        return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]])
    phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
    rho = np.outer(phi, phi)
    I2 = np.eye(2)
    # depolarizing on Bob (qubit 1 = left tensor factor in Qiskit order -> build with kron(B, A))
    ptB = rho.reshape(2, 2, 2, 2).trace(axis1=0, axis2=2)  # trace out Bob -> Alice's reduced (ordering: index = 2*qB + qA)
    rho = (1 - lam) * rho + lam * np.kron(I2 / 2, ptB)

    def obs(t):
        return math.cos(t) * np.diag([1, -1]) + math.sin(t) * np.array([[0, 1], [1, 0]])

    def E_of(r):
        return float(np.real(np.trace(r @ np.kron(obs(b), obs(a)))))

    e0 = E_of(rho)
    if p == 0:
        return e0
    ee = 0.0
    for e in EVE:
        r_new = np.zeros((4, 4))
        for m in (0, 1):
            v = ry(e) @ np.eye(2)[m]
            P = np.kron(np.outer(v, v), I2)  # project Bob onto eigenstate, keep it (resend)
            r_new += P @ rho @ P
        ee += E_of(r_new) / 3
    return (1 - p) * e0 + p * ee


# ------------------------------------------------------------- T1
@test("T1 Bell pair", 8)
def t1():
    c = g("create_bell_pair")()
    assert c.num_qubits == 2 and c.num_clbits == 3, "use 2 qubits and 3 classical bits"
    assert "measure" not in c.count_ops(), "no measurements allowed"
    s = Statevector(c)
    assert abs(abs(s.inner(Statevector(np.array([1, 0, 0, 1]) / math.sqrt(2)))) - 1) < 1e-9, "state is not |Phi+>"


# ------------------------------------------------------------- T2
@test("T2 measurement along theta (random angles, both qubits)", 10)
def t2():
    for theta in list(RNG.uniform(-10, 10, 6)) + [0, math.pi / 2, 2 * math.pi + 0.3, -1.0]:
        for q in (0, 1):
            for m in (0, 1):
                c = QuantumCircuit(2, 3)
                if m:
                    c.x(q)
                c.ry(theta, q)
                g("add_measurement")(c, q, q, theta)
                counts = SIM.run(c, shots=100).result().get_counts()
                bits = {k[-1 - q] for k in counts}
                assert bits == {str(m)}, f"theta={theta:.3f}, qubit {q}, eigenbit {m}: got {bits}"
    assert g("add_measurement")(QuantumCircuit(2, 3), 0, 0, 0.1) is None, "must return None"


# ------------------------------------------------------------- T3
@test("T3 intercept-resend: Eve's bit and resent state", 10)
def t3():
    for theta in RNG.uniform(-4, 4, 5):
        c = QuantumCircuit(2, 3)
        c.h(1)
        g("add_intercept_resend")(c, 1, 2, theta)
        ops = set(c.count_ops())
        assert "reset" not in ops, "reset is not allowed"
        assert not any(getattr(i.operation, "condition", None) for i in c.data), "no classical control allowed"
        g("add_measurement")(c, 1, 1, theta)
        counts = SIM.run(c, shots=400).result().get_counts()
        for k in counts:
            assert k[-2] == k[-3], f"theta={theta:.2f}: Bob along Eve's direction must reproduce Eve's bit"
        if abs(math.sin(theta)) < 0.9:
            assert len({k[-3] for k in counts}) == 2, "Eve's bit should be random for |+>"
        # resent state should be eigenstate of A(theta): measuring along theta+pi/2 must be 50/50
        c2 = QuantumCircuit(2, 3)
        g("add_intercept_resend")(c2, 0, 2, theta)
        g("add_measurement")(c2, 0, 0, theta)
        k2 = SIM.run(c2, shots=400).result().get_counts()
        for k in k2:
            assert k[-1] == k[-3], f"theta={theta:.2f}: resent state is not the eigenstate Eve observed"


# ------------------------------------------------------------- T4
def sample_E(c, shots=6000):
    mem = SIM.run(c, shots=shots, memory=True).result().get_memory()
    a = np.array([1 - 2 * int(m[-1]) for m in mem])
    b = np.array([1 - 2 * int(m[-2]) for m in mem])
    return float(np.mean(a * b)), shots


@test("T4 round circuit: correlations with/without Eve and channel", 8)
def t4():
    build = g("build_round_circuit")
    for _ in range(4):
        a, b = RNG.uniform(-3, 3, 2)
        c = build(a, b)
        assert c.num_qubits == 2 and c.num_clbits == 3, "2 qubits / 3 clbits"
        E, n = sample_E(c)
        assert abs(E - math.cos(a - b)) < 5 * math.sqrt(1 / n), f"no Eve: E={E:.3f} vs {math.cos(a-b):.3f}"
    lam = 0.4
    ch = g("transmission_channel")(lam)
    a, b = 0.2, 0.9
    E, n = sample_E(build(a, b, None, ch))
    assert abs(E - (1 - lam) * math.cos(a - b)) < 5 * math.sqrt(1 / n), f"channel: E={E:.3f}"
    e = EVE[1]
    E, n = sample_E(build(a, b, e))
    exp = math.cos(a - e) * math.cos(e - b)
    assert abs(E - exp) < 5 * math.sqrt(1 / n), f"Eve at {e:.3f}: E={E:.3f} vs {exp:.3f}"


# ------------------------------------------------------------- T5
@test("T5 correlation / CHSH / standard error (+ edge cases)", 12)
def t5():
    corr, chsh, se = g("correlation"), g("chsh_value"), g("chsh_standard_error")
    for _ in range(5):
        x = RNG.choice([-1, 1], 50)
        y = RNG.choice([-1, 1], 50)
        assert abs(corr(x, y) - np.mean(x * y)) < 1e-12
        assert abs(corr(list(x), list(y)) - np.mean(x * y)) < 1e-12, "must accept lists"
    for bad in (([], []), ([1, 1], [1]), ([1, 0], [1, 1]), ([1, 2], [1, 1])):
        expect_value_error(corr, *bad)
    for _ in range(5):
        Ed = {(i, j): float(RNG.uniform(-1, 1)) for i in range(3) for j in range(3)}
        cnt = {(i, j): int(RNG.integers(1, 2000)) for i in range(3) for j in range(3)}
        S = abs(sum(s * Ed[p] for p, s in CHSH))
        assert abs(chsh(Ed) - S) < 1e-9, f"S={chsh(Ed)}, expected {S}"
        sig = math.sqrt(sum((1 - Ed[p] ** 2) / cnt[p] for p, _ in CHSH))
        assert abs(se(Ed, cnt) - sig) < 1e-9, f"sigma_S={se(Ed, cnt)}, expected {sig}"
    Ed[(1, 1)] = float("nan")  # unused pair: must still work
    chsh(Ed)
    Ed[(0, 2)] = float("nan")
    expect_value_error(chsh, Ed)
    Ed[(0, 2)] = 0.5
    cnt[(2, 2)] = 0
    expect_value_error(se, Ed, cnt)


# ------------------------------------------------------------- T6
@test("T6 sifting and QBER", 10)
def t6():
    RS, RO = ns["RoundSettings"], ns["RoundOutcomes"]
    n = 500
    ac, bc = RNG.integers(0, 3, n), RNG.integers(0, 3, n)
    ab, bb = RNG.integers(0, 2, n), RNG.integers(0, 2, n)
    ev = np.where(RNG.random(n) < 0.3, RNG.integers(0, 3, n), -1)
    s, o = RS(ac, bc, ev, None), RO(ab, bb, np.where(ev >= 0, 0, -1))
    ka, kb = g("sift_key")(s, o)
    mask = ((ac == 1) & (bc == 0)) | ((ac == 2) & (bc == 1))
    assert np.array_equal(np.asarray(ka), ab[mask]) and np.array_equal(np.asarray(kb), bb[mask]), "wrong rounds or order"
    assert np.issubdtype(np.asarray(ka).dtype, np.integer), "keys must be integer arrays"
    q = g("qber")(ka, kb)
    assert abs(q - np.mean(ab[mask] != bb[mask])) < 1e-12
    expect_value_error(g("qber"), [], [])
    expect_value_error(g("qber"), [0, 1], [0])


# ------------------------------------------------------------- T7
@test("T7 analytic prediction vs exact density-matrix model", 8)
def t7():
    f = g("predicted_correlation")
    for _ in range(20):
        a, b = RNG.uniform(-4, 4, 2)
        p, lam = RNG.uniform(0, 1, 2)
        exp = exact_E(a, b, p, lam)
        got = f(a, b, p, lam)
        assert abs(got - exp) < 1e-9, f"a={a:.2f} b={b:.2f} p={p:.2f} lam={lam:.2f}: {got:.6f} vs {exp:.6f}"
    assert abs(f(0.3, 0.3) - 1.0) < 1e-12, "default arguments must give cos(a-b)"


# ------------------------------------------------------------- T8
@test("T8 run_e91 end-to-end vs predictions (5 sigma)", 16)
def t8():
    run = g("run_e91")
    keys = {"settings", "outcomes", "correlations", "counts", "S", "S_std_error",
            "key_alice", "key_bob", "qber", "key_length"}
    for n, p, lam, seed in ((9000, 0.0, 0.0, 3), (9000, 1.0, 0.0, 4), (12000, 0.5, 0.0, 5), (9000, 0.0, 0.3, 6)):
        ch = g("transmission_channel")(lam) if lam else None
        out = run(n, p, seed, ch)
        assert set(out) == keys, f"keys differ: {set(out) ^ keys}"
        ref = g("generate_round_settings")(n, p, seed)
        assert np.array_equal(out["settings"].alice_choices, ref.alice_choices), "settings not from given seed"
        assert isinstance(out["key_length"], int) and out["key_length"] == len(out["key_alice"]), "key_length"
        assert sum(out["counts"].values()) == n, "counts must sum to n_rounds"
        Ep = {(i, j): exact_E(A[i], B[j], p, lam) for i in range(3) for j in range(3)}
        S_pred = abs(sum(s * Ep[q] for q, s in CHSH))
        assert abs(out["S"] - S_pred) < 5 * out["S_std_error"], f"p={p} lam={lam}: S={out['S']:.3f} vs {S_pred:.3f}"
        nk = {kp: out["counts"][kp] for kp in KEY_PAIRS}
        q_pred = sum(nk[kp] * (1 - Ep[kp]) / 2 for kp in KEY_PAIRS) / max(1, sum(nk.values()))
        q_se = math.sqrt(max(q_pred * (1 - q_pred), 1e-4) / max(1, out["key_length"]))
        assert abs(out["qber"] - q_pred) < 5 * q_se, f"p={p}: QBER={out['qber']:.3f} vs {q_pred:.3f}"
    small = run(20, 0.0, 1)
    assert any(math.isnan(v) for k, v in small["correlations"].items() if small["counts"][k] == 0) or \
        all(v > 0 for v in small["counts"].values()), "empty pairs must give nan"


# ------------------------------------------------------------- T9
@test("T9 classifier on 9 hidden-style datasets", 8)
def t9():
    run, clf = g("run_e91"), g("classify_disturbance")
    score, rows = 0.0, []
    for kind, p, lam in (("eavesdropper", 1.0, 0.0), ("depolarizing", 0.0, 0.4), ("eavesdropper", 0.6, 0.2)):
        for seed in (21, 22, 23):
            ch = g("transmission_channel")(lam) if lam else None
            out = run(30000, p, seed, ch)
            lab = clf(out["correlations"], out["counts"])
            assert lab in ("eavesdropper", "depolarizing", "inconclusive"), f"bad label {lab!r}"
            score += 1.0 if lab == kind else (1 / 3 if lab == "inconclusive" else 0.0)
            rows.append(f"{kind[:5]}(p={p},lam={lam}) -> {lab}")
    print("      " + "; ".join(rows))
    frac = score / 9
    RESULTS_PARTIAL["T9"] = 8 * frac
    assert frac > 0.999, f"classifier score {frac:.2f} (partial credit {8*frac:.1f}/8 counted)"


RESULTS_PARTIAL = {}

if __name__ == "__main__":
    earned = 0.0
    for name, pts, fn in RESULTS:
        try:
            fn()
            earned += pts
            print(f"PASS  [{pts:>2}] {name}")
        except NotImplementedError:
            print(f"TODO  [{pts:>2}] {name}: not implemented")
        except AssertionError as e:
            earned += RESULTS_PARTIAL.get(name[:2], 0)
            print(f"FAIL  [{pts:>2}] {name}: {e}")
        except Exception as e:
            print(f"ERROR [{pts:>2}] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\nEstimated automatic score: {earned:.1f}/90")
