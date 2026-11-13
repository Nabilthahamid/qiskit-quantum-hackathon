"""Public helper utilities for the Qiskit Quantum Hackathon notebooks.

Organizer-provided code. Do not modify this file. The official grader always
uses its own canonical copy, so local edits have no effect on grading and may
make your local results disagree with the official evaluation.

The module is organised by challenge:

* Common utilities: environment checks, seeds, official transpilation.
* Challenge 1 (E91): measurement settings, round generation, round execution.
* Challenge 2 (Grover): shot-counted circuit execution.
* Challenge 3 (VQE): public molecular data, official noise model, budget-enforcing
  sampler.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HELPERS_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

#: Package versions used to build and verify the hackathon material.
TESTED_VERSIONS: Dict[str, str] = {
    "qiskit": "2.6.1",
    "qiskit_aer": "0.18.2",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "matplotlib": "3.11.2",
}

#: Basis gate set used by the official transpilation pipeline.
BASIS_GATES: Tuple[str, ...] = ("rz", "sx", "x", "cx")

#: Fixed transpiler seed so that gate counts are reproducible.
TRANSPILE_SEED = 1234

#: Optimization level of the official transpilation pipeline.
TRANSPILE_OPTIMIZATION_LEVEL = 1

DEFAULT_SEED = 2026


def get_seed(default: int = DEFAULT_SEED) -> int:
    """Return the session seed.

    The grader sets the environment variable ``QHACK_SEED`` to a fresh value for
    every official run, so results must not depend on one particular seed.
    """
    value = os.environ.get("QHACK_SEED")
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def check_environment(verbose: bool = True) -> Dict[str, Optional[str]]:
    """Report installed versions of the required packages.

    Returns a dictionary mapping package name to installed version (or ``None``
    when the package is missing). Version differences produce a warning, not an
    error, because minor releases are usually compatible.
    """
    found: Dict[str, Optional[str]] = {}
    for name, tested in TESTED_VERSIONS.items():
        try:
            module = importlib.import_module(name)
            found[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            found[name] = None
    if verbose:
        print(f"public_helpers version: {HELPERS_VERSION}")
        for name, tested in TESTED_VERSIONS.items():
            installed = found[name]
            if installed is None:
                status = "MISSING"
            elif installed == tested:
                status = "ok"
            else:
                status = f"differs from tested version {tested}"
            print(f"  {name:<11} {str(installed):<10} {status}")
    missing = [name for name, version in found.items() if version is None]
    if missing:
        raise ImportError(
            "Missing required packages: " + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )
    return found


def official_transpile(circuit):
    """Transpile ``circuit`` with the official, fixed settings.

    The official pipeline uses the basis ``rz, sx, x, cx``, all-to-all
    connectivity, optimization level 1 and a fixed transpiler seed. Gate counts
    reported by the grader are always measured on the output of this function.
    """
    from qiskit import transpile

    return transpile(
        circuit,
        basis_gates=list(BASIS_GATES),
        optimization_level=TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=TRANSPILE_SEED,
    )


def count_two_qubit_gates(circuit, transpile_first: bool = True) -> int:
    """Count gates acting on two or more qubits.

    When ``transpile_first`` is true the circuit is first passed through
    :func:`official_transpile`, which is how resource budgets are measured.
    """
    target = official_transpile(circuit) if transpile_first else circuit
    count = 0
    for instruction in target.data:
        operation = instruction.operation
        if operation.name in ("barrier", "measure", "reset", "delay"):
            continue
        if len(instruction.qubits) >= 2:
            count += 1
    return count


def run_public_check(label: str, check: Callable[[], None]) -> bool:
    """Run a public check and print a one-line status.

    Public checks only test basic properties (types, shapes, simple physical
    facts stated in the notebook). Passing them does not guarantee full marks:
    the official grader uses additional hidden tests.
    """
    try:
        check()
    except NotImplementedError:
        print(f"[{label}] not implemented yet")
        return False
    except AssertionError as error:
        print(f"[{label}] FAILED: {error}")
        return False
    except Exception as error:
        print(f"[{label}] ERROR: {type(error).__name__}: {error}")
        return False
    print(f"[{label}] passed")
    return True


def _derive_seed(seed: Optional[int], *labels: object) -> int:
    """Derive a reproducible 32-bit seed from a base seed and labels."""
    text = json.dumps([seed, *[str(label) for label in labels]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


# ---------------------------------------------------------------------------
# Challenge 1: E91 quantum key distribution
# ---------------------------------------------------------------------------

#: Alice's measurement directions (angles in the x-z plane of the Bloch sphere).
ALICE_ANGLES: Tuple[float, float, float] = (0.0, math.pi / 4, math.pi / 2)

#: Bob's measurement directions.
BOB_ANGLES: Tuple[float, float, float] = (math.pi / 4, math.pi / 2, 3 * math.pi / 4)

#: Directions from which an intercept-resend eavesdropper chooses uniformly.
EVE_ANGLES: Tuple[float, float, float] = BOB_ANGLES

#: Classical bit layout of every E91 round circuit.
E91_CLBIT_ALICE = 0
E91_CLBIT_BOB = 1
E91_CLBIT_EVE = 2


@dataclass(frozen=True)
class RoundSettings:
    """Random choices made in every round of the E91 protocol.

    Attributes
    ----------
    alice_choices, bob_choices:
        Integer arrays with values in ``{0, 1, 2}``, indices into
        :data:`ALICE_ANGLES` and :data:`BOB_ANGLES`.
    eve_choices:
        Integer array with values in ``{-1, 0, 1, 2}``. The value ``-1`` means
        that Eve did not intercept the round; otherwise it is an index into
        :data:`EVE_ANGLES`.
    seed:
        Seed used to generate the settings.
    """

    alice_choices: np.ndarray
    bob_choices: np.ndarray
    eve_choices: np.ndarray
    seed: Optional[int]

    @property
    def n_rounds(self) -> int:
        return int(self.alice_choices.size)

    @property
    def intercepted(self) -> np.ndarray:
        return self.eve_choices >= 0


@dataclass(frozen=True)
class RoundOutcomes:
    """Measurement bits (0 or 1) recorded in every round."""

    alice_bits: np.ndarray
    bob_bits: np.ndarray
    eve_bits: np.ndarray  # -1 where Eve did not intercept


def generate_round_settings(n_rounds: int, p_intercept: float = 0.0,
                            seed: Optional[int] = None) -> RoundSettings:
    """Generate the random basis choices for ``n_rounds`` rounds.

    Alice and Bob each choose one of their three directions independently and
    uniformly at random. Each round is intercepted independently with
    probability ``p_intercept``; an intercepting Eve chooses one of
    :data:`EVE_ANGLES` uniformly at random. For a fixed seed, Alice's and Bob's
    choices do not depend on ``p_intercept``.
    """
    if int(n_rounds) != n_rounds or n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer")
    if not 0.0 <= float(p_intercept) <= 1.0:
        raise ValueError("p_intercept must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    n_rounds = int(n_rounds)
    alice = rng.integers(0, 3, size=n_rounds)
    bob = rng.integers(0, 3, size=n_rounds)
    intercept_draw = rng.random(n_rounds)
    eve_basis = rng.integers(0, 3, size=n_rounds)
    eve = np.where(intercept_draw < float(p_intercept), eve_basis, -1)
    return RoundSettings(alice.astype(int), bob.astype(int), eve.astype(int), seed)


def transmission_channel(depolarizing_lambda: float):
    """Return a single-qubit depolarizing channel for the transmission line.

    The channel maps :math:`\\rho \\mapsto (1-\\lambda)\\rho + \\lambda I/2`.
    The returned object can be appended to a circuit, for example
    ``qc.append(channel, [1])``.
    """
    from qiskit_aer.noise import depolarizing_error

    if not 0.0 <= float(depolarizing_lambda) <= 1.0:
        raise ValueError("depolarizing_lambda must lie in [0, 1]")
    return depolarizing_error(float(depolarizing_lambda), 1)


def execute_rounds(settings: RoundSettings,
                   build_round_circuit: Callable[..., object],
                   seed: Optional[int] = None,
                   channel=None) -> RoundOutcomes:
    """Simulate every round of the protocol with Qiskit Aer.

    Rounds that share the same (Alice, Bob, Eve) settings are simulated with a
    single circuit and one shot per round, which is statistically equivalent to
    simulating every round separately because rounds are independent.

    Parameters
    ----------
    settings:
        Output of :func:`generate_round_settings`.
    build_round_circuit:
        Your implementation of ``build_round_circuit(alice_angle, bob_angle,
        eve_angle=None, channel=None)``.
    seed:
        Simulator seed. The same seed and settings reproduce the same outcomes.
    channel:
        Optional transmission channel passed through to ``build_round_circuit``.
    """
    from qiskit_aer import AerSimulator

    n_rounds = settings.n_rounds
    alice_bits = np.full(n_rounds, -1, dtype=int)
    bob_bits = np.full(n_rounds, -1, dtype=int)
    eve_bits = np.full(n_rounds, -1, dtype=int)
    groups: Dict[Tuple[int, int, int], np.ndarray] = {}
    keys = np.stack([settings.alice_choices, settings.bob_choices, settings.eve_choices], axis=1)
    for key in sorted(set(map(tuple, keys.tolist()))):
        mask = np.all(keys == np.array(key), axis=1)
        groups[tuple(int(k) for k in key)] = np.flatnonzero(mask)
    simulator = AerSimulator()
    for (alice_index, bob_index, eve_index), rounds in groups.items():
        eve_angle = None if eve_index < 0 else EVE_ANGLES[eve_index]
        circuit = build_round_circuit(
            ALICE_ANGLES[alice_index], BOB_ANGLES[bob_index], eve_angle, channel
        )
        if circuit.num_clbits < 2:
            raise ValueError("round circuits must have at least two classical bits")
        group_seed = _derive_seed(seed, "e91", alice_index, bob_index, eve_index)
        result = simulator.run(circuit, shots=int(rounds.size), memory=True,
                               seed_simulator=group_seed).result()
        memory = result.get_memory(circuit)
        for position, bitstring in zip(rounds, memory):
            bits = bitstring.replace(" ", "")
            alice_bits[position] = int(bits[-1 - E91_CLBIT_ALICE])
            bob_bits[position] = int(bits[-1 - E91_CLBIT_BOB])
            if eve_index >= 0 and len(bits) > E91_CLBIT_EVE:
                eve_bits[position] = int(bits[-1 - E91_CLBIT_EVE])
    return RoundOutcomes(alice_bits, bob_bits, eve_bits)


def bits_to_values(bits: Iterable[int]) -> np.ndarray:
    """Convert measurement bits to eigenvalues: bit 0 gives +1 and bit 1 gives -1."""
    array = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=int)
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError("bits must contain only 0 and 1")
    return 1 - 2 * array


def plot_correlation_table(correlations: Dict[Tuple[int, int], float], title: str = "",
                           ax=None):
    """Plot the 3 x 3 table of correlation values E(a_i, b_j)."""
    import matplotlib.pyplot as plt

    table = np.full((3, 3), np.nan)
    for (i, j), value in correlations.items():
        table[i, j] = value
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(table, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(3):
        for j in range(3):
            if not np.isnan(table[i, j]):
                ax.text(j, i, f"{table[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), [f"b{j + 1}" for j in range(3)])
    ax.set_yticks(range(3), [f"a{i + 1}" for i in range(3)])
    ax.set_xlabel("Bob's setting")
    ax.set_ylabel("Alice's setting")
    ax.set_title(title or "Correlation table E(a_i, b_j)")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ---------------------------------------------------------------------------
# Challenge 2: Grover search
# ---------------------------------------------------------------------------

@dataclass
class ShotLedger:
    """Counts shots and circuit executions made through :func:`run_counts`."""

    shots: int = 0
    executions: int = 0

    def reset(self) -> None:
        self.shots = 0
        self.executions = 0


#: Global ledger updated by :func:`run_counts`.
SHOT_LEDGER = ShotLedger()


def run_counts(circuit, shots: int, seed: Optional[int] = None, noise_model=None) -> Dict[str, int]:
    """Run ``circuit`` on Qiskit Aer and return measurement counts.

    The circuit is first passed through :func:`official_transpile`, so that
    noise models defined on the basis gates ``rz, sx, x, cx`` act on every gate.
    Every call is recorded in :data:`SHOT_LEDGER`, which the grader uses to
    verify shot budgets.
    """
    from qiskit_aer import AerSimulator

    if int(shots) != shots or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if circuit.num_clbits == 0:
        raise ValueError("the circuit has no classical bits to measure into")
    compiled = official_transpile(circuit)
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = simulator.run(compiled, shots=int(shots)).result()
    SHOT_LEDGER.shots += int(shots)
    SHOT_LEDGER.executions += 1
    counts = result.get_counts(compiled)
    return {key.replace(" ", ""): int(value) for key, value in counts.items()}


# ---------------------------------------------------------------------------
# Challenge 3: VQE under a resource budget
# ---------------------------------------------------------------------------

#: Location of the public molecular dataset, relative to the repository root.
PUBLIC_VQE_DATA = Path("data") / "lih_sto3g_active_space_public.json"

#: Two-qubit depolarizing parameter of the official VQE noise model.
VQE_TWO_QUBIT_DEPOLARIZING = 0.002

#: Single-qubit depolarizing parameter of the official VQE noise model.
VQE_ONE_QUBIT_DEPOLARIZING = 0.0002

#: Chemical accuracy in hartree.
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


@dataclass(frozen=True)
class VQEBudget:
    """Resource budget for one call of ``solve_geometry``.

    All limits are enforced by :class:`BudgetedSampler` and verified again by the
    grader.
    """

    max_qubits: int = 6
    max_two_qubit_gates: int = 40
    max_jobs: int = 600
    max_circuit_executions: int = 20_000
    max_total_shots: int = 20_000_000
    max_runtime_seconds: int = 300


VQE_BUDGET = VQEBudget()


class BudgetExceededError(RuntimeError):
    """Raised when a request would exceed the official resource budget."""


def find_repository_root(start: Optional[Path] = None) -> Path:
    """Locate the directory that contains ``src/public_helpers.py``."""
    env_root = os.environ.get("QHACK_REPO_ROOT")
    if env_root:
        return Path(env_root)
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "public_helpers.py").exists():
            return candidate
    module_root = Path(__file__).resolve().parent.parent
    if (module_root / "src" / "public_helpers.py").exists():
        return module_root
    raise FileNotFoundError("Could not locate the hackathon repository root.")


def load_public_vqe_dataset() -> dict:
    """Load the public LiH active-space dataset (integrals and references)."""
    path = find_repository_root() / PUBLIC_VQE_DATA
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_public_integrals(bond_length: Optional[float] = None):
    """Return integrals for one public geometry, or all of them.

    Each entry is a dictionary with the keys ``bond_length_angstrom``,
    ``n_orbitals``, ``n_electrons``, ``h1`` (shape ``(n, n)``), ``eri`` (shape
    ``(n, n, n, n)``, chemists' notation) and ``e_core`` (hartree). Arrays are
    returned as NumPy arrays. Public entries also contain ``e_hf`` and
    ``e_ref`` (the exact active-space reference energy).
    """
    dataset = load_public_vqe_dataset()
    entries = []
    for entry in dataset["geometries"]:
        item = dict(entry)
        item["h1"] = np.asarray(item["h1"], dtype=float)
        item["eri"] = np.asarray(item["eri"], dtype=float)
        entries.append(item)
    if bond_length is None:
        return entries
    for item in entries:
        if abs(item["bond_length_angstrom"] - float(bond_length)) < 1e-9:
            return item
    available = ", ".join(f"{item['bond_length_angstrom']:.2f}" for item in entries)
    raise KeyError(f"No public data for bond length {bond_length}. Available: {available}")


_NOISE_MODEL_CACHE: Dict[str, object] = {}


def vqe_noise_model():
    """Official noise model of Challenge 3.

    Two-qubit depolarizing error with parameter 0.002 on every ``cx`` gate,
    single-qubit depolarizing error with parameter 0.0002 on every ``sx`` and
    ``x`` gate, and no error on ``rz`` (implemented as a virtual frame change
    on IBM hardware). Measurement basis changes and readout are ideal.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    if "vqe" not in _NOISE_MODEL_CACHE:
        model = NoiseModel(basis_gates=list(BASIS_GATES))
        model.add_all_qubit_quantum_error(depolarizing_error(VQE_ONE_QUBIT_DEPOLARIZING, 1), ["sx", "x"])
        model.add_all_qubit_quantum_error(depolarizing_error(VQE_TWO_QUBIT_DEPOLARIZING, 2), ["cx"])
        _NOISE_MODEL_CACHE["vqe"] = model
    return _NOISE_MODEL_CACHE["vqe"]


_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
_SDG = np.diag([1.0, -1.0j])
_BASIS_ROTATIONS = {
    "Z": None,
    "I": None,
    "X": _HADAMARD,
    "Y": _HADAMARD @ _SDG,
}


def basis_probabilities(density_matrix: np.ndarray, basis: str) -> np.ndarray:
    """Outcome probabilities when every qubit is measured in ``basis``.

    ``basis`` uses Qiskit's ordering: the last character refers to qubit 0.
    The characters ``I`` and ``Z`` mean a computational-basis measurement, ``X``
    applies a Hadamard gate before measuring and ``Y`` applies
    :math:`S^\\dagger` followed by a Hadamard gate. The returned array is
    indexed by the integer value of the measured bit string.
    """
    n_qubits = len(basis)
    tensor = np.asarray(density_matrix, dtype=complex).reshape((2,) * (2 * n_qubits))
    for qubit in range(n_qubits):
        rotation = _BASIS_ROTATIONS[basis[n_qubits - 1 - qubit]]
        if rotation is None:
            continue
        row_axis = n_qubits - 1 - qubit
        col_axis = 2 * n_qubits - 1 - qubit
        tensor = np.moveaxis(np.tensordot(rotation, tensor, axes=([1], [row_axis])), 0, row_axis)
        tensor = np.moveaxis(np.tensordot(rotation.conj(), tensor, axes=([1], [col_axis])), 0, col_axis)
    matrix = tensor.reshape(2 ** n_qubits, 2 ** n_qubits)
    probabilities = np.clip(np.real(np.diag(matrix)), 0.0, None)
    return probabilities / probabilities.sum()


def _circuit_fingerprint(circuit) -> str:
    parts = [str(circuit.num_qubits)]
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        params = [str(p) for p in operation.params]
        parts.append(f"{operation.name}|{qubits}|{params}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def noisy_density_matrix(circuit, parameter_values: Optional[Sequence[float]] = None) -> np.ndarray:
    """Exact density matrix produced by ``circuit`` under :func:`vqe_noise_model`.

    The circuit is transpiled with :func:`official_transpile` before the noisy
    density-matrix simulation. This function is used by the sampler and by the
    grader; calling it directly inside ``solve_geometry`` is not permitted
    because it bypasses shot noise and the budget.
    """
    from qiskit_aer import AerSimulator

    compiled = official_transpile(circuit)
    if parameter_values is not None and compiled.num_parameters:
        compiled = compiled.assign_parameters(_ordered_binding(compiled, circuit, parameter_values))
    compiled.save_density_matrix()
    simulator = AerSimulator(method="density_matrix", noise_model=vqe_noise_model())
    data = simulator.run(compiled, shots=1).result().data()
    return np.asarray(data["density_matrix"])


def _ordered_binding(compiled, original, parameter_values):
    values = np.asarray(parameter_values, dtype=float).ravel()
    parameters = list(original.parameters)
    if len(parameters) != values.size:
        raise ValueError(
            f"expected {len(parameters)} parameter values, received {values.size}"
        )
    mapping = dict(zip(parameters, values))
    return {parameter: mapping[parameter] for parameter in compiled.parameters}


@dataclass
class LedgerEntry:
    """One sampler job as recorded in :attr:`BudgetedSampler.ledger`."""

    job: int
    fingerprint: str
    num_qubits: int
    two_qubit_gates: int
    parameters: List[float]
    bases: List[str]
    shots: int


@dataclass
class BudgetedSampler:
    """Noisy, shot-based sampler that enforces the official VQE budget.

    Every call to :meth:`run` is one *job*. A job prepares the state of a
    parameterized circuit (no measurements) under :func:`vqe_noise_model`,
    measures it in one or more Pauli bases and returns one counts dictionary per
    basis. Each basis counts as one circuit execution, and each execution uses
    ``shots`` shots.

    Use one sampler per geometry. Requests that would exceed the budget raise
    :class:`BudgetExceededError` before anything is simulated.
    """

    budget: VQEBudget = field(default_factory=VQEBudget)
    seed: Optional[int] = None
    jobs: int = 0
    circuit_executions: int = 0
    total_shots: int = 0
    ledger: List[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        from qiskit_aer import AerSimulator

        self._rng = np.random.default_rng(self.seed)
        self._cache: Dict[str, Tuple[object, int]] = {}
        self._start_time = time.monotonic()
        self._simulator = AerSimulator(method="density_matrix", noise_model=vqe_noise_model())

    @property
    def usage(self) -> Dict[str, int]:
        """Resources consumed so far."""
        return {
            "jobs": self.jobs,
            "circuit_executions": self.circuit_executions,
            "total_shots": self.total_shots,
        }

    @property
    def remaining(self) -> Dict[str, int]:
        """Resources still available."""
        return {
            "jobs": self.budget.max_jobs - self.jobs,
            "circuit_executions": self.budget.max_circuit_executions - self.circuit_executions,
            "total_shots": self.budget.max_total_shots - self.total_shots,
        }

    def _compile(self, circuit):
        fingerprint = _circuit_fingerprint(circuit)
        if fingerprint not in self._cache:
            for instruction in circuit.data:
                name = instruction.operation.name
                if name in ("measure", "reset") or instruction.clbits:
                    raise ValueError(
                        "sampler circuits must not contain measurements, resets or classical "
                        "operations; the sampler measures in the requested bases"
                    )
            if circuit.num_qubits > self.budget.max_qubits:
                raise BudgetExceededError(
                    f"circuit uses {circuit.num_qubits} qubits; the limit is {self.budget.max_qubits}"
                )
            compiled = official_transpile(circuit)
            two_qubit = count_two_qubit_gates(compiled, transpile_first=False)
            if two_qubit > self.budget.max_two_qubit_gates:
                raise BudgetExceededError(
                    f"circuit has {two_qubit} two-qubit gates after official transpilation; "
                    f"the limit is {self.budget.max_two_qubit_gates}"
                )
            self._cache[fingerprint] = (compiled, two_qubit)
        compiled, two_qubit = self._cache[fingerprint]
        return fingerprint, compiled, two_qubit

    def run(self, circuit, parameter_values: Optional[Sequence[float]] = None,
            bases: Sequence[str] = (), shots: int = 1000) -> List[Dict[str, int]]:
        """Prepare ``circuit`` and measure it in every basis of ``bases``.

        Parameters
        ----------
        circuit:
            ``QuantumCircuit`` without measurements. It may contain parameters.
        parameter_values:
            Values for ``circuit.parameters`` in the order of that property
            (Qiskit sorts parameters by name). Use ``None`` for circuits
            without parameters.
        bases:
            Sequence of Pauli basis strings of length ``circuit.num_qubits``,
            in Qiskit order (the last character is qubit 0). Allowed characters
            are ``X``, ``Y``, ``Z`` and ``I``.
        shots:
            Shots per basis.

        Returns
        -------
        list of dict
            One counts dictionary per basis, with bit strings in Qiskit order.
        """
        if isinstance(bases, str):
            bases = [bases]
        bases = list(bases)
        if not bases:
            raise ValueError("provide at least one measurement basis")
        if int(shots) != shots or shots <= 0:
            raise ValueError("shots must be a positive integer")
        shots = int(shots)
        for basis in bases:
            if len(basis) != circuit.num_qubits or any(ch not in "IXYZ" for ch in basis):
                raise ValueError(
                    f"invalid basis {basis!r}: expected {circuit.num_qubits} characters from IXYZ"
                )
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.budget.max_runtime_seconds:
            raise BudgetExceededError(
                f"runtime limit of {self.budget.max_runtime_seconds} s exceeded"
            )
        if self.jobs + 1 > self.budget.max_jobs:
            raise BudgetExceededError(f"job limit of {self.budget.max_jobs} reached")
        if self.circuit_executions + len(bases) > self.budget.max_circuit_executions:
            raise BudgetExceededError(
                f"circuit-execution limit of {self.budget.max_circuit_executions} would be exceeded"
            )
        if self.total_shots + shots * len(bases) > self.budget.max_total_shots:
            raise BudgetExceededError(
                f"shot limit of {self.budget.max_total_shots} would be exceeded"
            )
        fingerprint, compiled, two_qubit = self._compile(circuit)
        values: List[float] = []
        bound = compiled
        if circuit.num_parameters:
            if parameter_values is None:
                raise ValueError("parameter_values are required for a parameterized circuit")
            bound = compiled.assign_parameters(
                _ordered_binding(compiled, circuit, parameter_values)
            )
            values = [float(v) for v in np.asarray(parameter_values, dtype=float).ravel()]
        bound = bound.copy()
        bound.save_density_matrix()
        density = np.asarray(self._simulator.run(bound, shots=1).result().data()["density_matrix"])
        n_qubits = circuit.num_qubits
        results: List[Dict[str, int]] = []
        for basis in bases:
            probabilities = basis_probabilities(density, basis)
            samples = self._rng.multinomial(shots, probabilities)
            results.append({
                format(index, f"0{n_qubits}b"): int(count)
                for index, count in enumerate(samples) if count
            })
        self.jobs += 1
        self.circuit_executions += len(bases)
        self.total_shots += shots * len(bases)
        self.ledger.append(LedgerEntry(self.jobs, fingerprint, n_qubits, two_qubit, values,
                                       list(bases), shots))
        return results


def plot_dissociation_curve(bond_lengths: Sequence[float], energies: Dict[str, Sequence[float]],
                            ax=None, title: str = "LiH dissociation curve"):
    """Plot several energy curves (hartree) against bond length (angstrom)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, values in energies.items():
        ax.plot(bond_lengths, values, marker="o", label=label)
    ax.set_xlabel("Li-H bond length (angstrom)")
    ax.set_ylabel("Energy (hartree)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax

