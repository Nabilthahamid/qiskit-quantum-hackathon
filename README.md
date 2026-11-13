# Qiskit Quantum Hackathon: Participant Guide

Welcome to the hackathon. This package contains three problems of increasing difficulty, each in its own Jupyter notebook. All problems run on simulators: you do not need access to quantum hardware, a cloud account or any paid service.

**Hackathon page:** https://ibm-qiskit-hackathon.devpost.com/

## Event schedule: Friday, 13 November 2026

| Time | Session |
|:--|:--|
| 10:00 AM | Notebooks released; participants start working from home |
| 5:00 to 5:30 PM | Informal talks with Figma and the IBM Social Club (in person) |
| 5:30 to 7:00 PM | Problem solving continues at the venue |
| 7:00 PM | Submission deadline |
| 7:00 to 8:00 PM | Panel discussion |

**Submission deadline:** 7:00 PM on Friday, 13 November 2026, at the end of the problem-solving session.
**Submission channel:** https://ibm-qiskit-hackathon.devpost.com/

You have about nine hours in total. Install the environment (Section 3) as soon as you receive the package, work from home during the day, and bring a laptop with your environment and your notebooks to the venue. You may attempt the challenges in any order, and partial solutions earn partial credit, so submit whatever you have finished by the deadline.

## 1. The three challenges

| Notebook | Challenge | Difficulty |
|:--|:--|:--|
| `notebooks/01_e91_quantum_key_distribution.ipynb` | E91 quantum key distribution with an eavesdropper | Easy | 
| `notebooks/02_grover_search_under_noise.ipynb` | Grover's algorithm for key search under noise | Medium | 
| `notebooks/03_vqe_molecular_energy.ipynb` | Variational quantum eigensolver for LiH under a resource budget | Hard |

**Challenge 1 (E91).** Build a simplified simulation of Ekert's entanglement-based key distribution protocol: prepare Bell pairs, measure along random directions, sift the key, evaluate the CHSH inequality, and detect an intercept-resend eavesdropper. A stretch task asks whether interception can be distinguished from depolarizing noise.

**Challenge 2 (Grover).** Implement Grover search for a hidden key, verify the ideal success probability, and measure how depolarizing and bit-flip noise change the success probability and the best number of iterations, compared with a fair classical baseline.

**Challenge 3 (VQE).** Build the qubit Hamiltonian of lithium hydride from precomputed molecular integrals, reduce it with symmetries, design an ansatz and run a noisy, shot-limited optimization within a fixed budget. Your workflow is evaluated at bond lengths that are not disclosed in advance. This challenge decides close rankings.

## 2. Required background

* Linear algebra with complex vectors and matrices; Dirac notation.
* Single- and two-qubit gates, measurement, Bell states.
* Python with NumPy; basic plotting with Matplotlib.
* For Challenge 2: density matrices and noise channels.
* For Challenge 3: second quantization (creation and annihilation operators) and the variational principle. The notebook summarizes what you need.

## 3. Installation

You need **Python 3.11 or newer** (several pinned packages require it). The notebooks were tested with Python 3.11.15 and the package versions pinned in `requirements.txt` (Qiskit 2.6.1, Qiskit Aer 0.18.2). Install them in a fresh environment.

### Linux and macOS

```bash
cd qiskit-quantum-hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd qiskit-quantum-hackathon
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

All required packages provide Windows wheels. The molecular integrals for Challenge 3 are included as a data file, so you do not need PySCF (which does not support Windows natively).

### Conda

```bash
conda env create -f environment.yml
conda activate qiskit-hackathon
```

### Google Colab

Colab works without a local installation. The organizers have not tested every Colab runtime, so keep a local copy of your work.

1. Open Colab and upload a notebook with *File*, *Upload notebook*.
2. Upload the ZIP archive of this package with the *Files* panel on the left.
3. Insert a new code cell at the very top of the notebook and run:

```python
!unzip -q -o qiskit-quantum-hackathon-participant-*.zip
%cd /content/qiskit-quantum-hackathon
!pip install -q -r requirements.txt
```

4. If Colab asks you to restart the session, do so and run the `%cd` line again. Then run the notebook from the beginning.
5. **Delete the extra cell before you submit**, and download the notebook with *File*, *Download*, *Download .ipynb*.

### Check the installation

```bash
python -c "import sys; sys.path.insert(0, 'src'); import public_helpers as ph; ph.check_environment()"
```

Every line should end with `ok`. A different minor version prints a warning; the grader always uses the pinned versions.

## 4. Working with the notebooks

Start Jupyter from the package folder, so that the notebooks can find `src/public_helpers.py`:

```bash
jupyter lab
```

### Cells you may and may not edit

* A cell whose first line is `# Provided Code: Do Not Modify` is organizer-provided. Do not edit, delete, duplicate or reorder it. In JupyterLab these cells are locked against editing.
* A cell whose first line is `# Participant Code: Task ...` is yours. Replace the line `# Your code here` and the `raise NotImplementedError(...)` statement with your implementation. Keep the function names and parameter lists exactly as given; you may add parameters with default values at the end.
* You may add new cells anywhere, for exploration, helper functions and plots. Helpers must be defined before they are used.
* Written answers go into the Markdown cells labelled **Your answer**.
* Do not edit `src/public_helpers.py` or the data file. The grader uses its own copies.

### Hints

Every task has at least three hints in collapsible sections labelled **Reveal Hint 1**, **Reveal Hint 2** and **Reveal Hint 3**. Click a label to open it. Hint 1 points to the relevant idea, Hint 2 gives the key mathematical relation, and Hint 3 suggests Qiskit tools. Using hints does not affect your score.

### Public checks

After each task a protected cell runs a short public check and prints `passed`, `not implemented yet` or a failure message. These checks test only basic properties. The official grader uses additional hidden tests with other inputs, so make sure that your functions implement the general behavior described in each specification.

### Reproducibility and seeds

Every notebook reads its random seed from the environment variable `QHACK_SEED` (default 2026). The grader runs your notebook with a different seed, so your code must work and your conclusions must hold for any seed.

### Allowed packages

Use only the Python standard library and the packages in `requirements.txt`. The grading environment contains nothing else and has no network access, so `pip install` inside a notebook fails there.

## 5. How your submission is graded

1. Your notebook is executed from top to bottom in a fresh kernel, in an isolated environment without network access, with a fresh seed. **Saved outputs in your file are ignored.**
2. If a cell raises an error, execution continues with the next cell, so that independent tasks can still earn points. Only the criteria that depend on the failing code are affected.
3. After execution, the grader calls your functions with hidden inputs (other keys, angles, sizes, noise levels, seeds and molecular geometries) and compares the results with exact or statistical expectations. Statistical checks use tolerances of five standard errors, so a correct implementation passes with overwhelming probability.
4. Scientific explanations and figures are assessed by judges with a structured review form that is identical for all participants.

Time limits for the complete execution in the grading environment: 15 minutes for Challenge 1, 20 minutes for Challenge 2 and 30 minutes for Challenge 3, plus the time of the hidden tests. In Challenge 3 every call of `solve_geometry` must also finish within 300 seconds; it is called twice for each undisclosed geometry, with independent seeds, and the scores are averaged.

## 6. Published scoring criteria

Each challenge is worth 100 points; the maximum total is 300.

### Challenge 1: E91 quantum key distribution

| Criterion | Points | Assessment |
|:--|--:|:--|
| E91-C1 Bell-state preparation (T1) | 8 | automatic |
| E91-C2 Measurement along a direction (T2) | 10 | automatic |
| E91-C3 Intercept-resend attack (T3) | 10 | automatic |
| E91-C4 Round circuit, channel and bit layout (T4) | 8 | automatic |
| E91-C5 Correlations, CHSH value and its standard error (T5) | 12 | automatic |
| E91-C6 Sifting and QBER (T6) | 10 | automatic |
| E91-C7 Analytic predictions (T7) | 8 | automatic |
| E91-C8 End-to-end protocol with hidden parameters and seeds (T8) | 16 | automatic |
| E91-C9 Stretch: interception versus noise (T9) | 8 | automatic |
| E91-C10 Scientific interpretation (Q1 to Q4, experiments and figures) | 10 | manual |

### Challenge 2: Grover search under noise

| Criterion | Points | Assessment |
|:--|--:|:--|
| GRV-C1 Oracle correctness for hidden keys (T1) | 15 | automatic |
| GRV-C2 Diffusion operator (T2) | 10 | automatic |
| GRV-C3 Grover circuit and ideal success probability (T3) | 15 | automatic |
| GRV-C4 Ideal theory: success probability and optimal iterations (T4) | 8 | automatic |
| GRV-C5 Classical baseline (T5) | 4 | automatic |
| GRV-C6 Depolarizing noise model (T6) | 8 | automatic |
| GRV-C7 Bit-flip noise model (T7) | 8 | automatic |
| GRV-C8 Noisy success-probability estimation (T8) | 10 | automatic |
| GRV-C9 Best iteration count under a shot budget (T9) | 12 | automatic |
| GRV-C10 Experimental design, resource accounting and interpretation | 10 | manual |

### Challenge 3: VQE under a resource budget

| Criterion | Points | Assessment |
|:--|--:|:--|
| VQE-C1 Hamiltonian construction and Jordan-Wigner mapping (T1) | 15 | automatic |
| VQE-C2 Symmetry reduction (T2) | 10 | automatic |
| VQE-C3 Measurement grouping (T3) | 8 | automatic |
| VQE-C4 Expectation values from counts (T4) | 7 | automatic |
| VQE-C5 Ansatz validity and two-qubit gate budget | 10 | automatic |
| VQE-C6 Budget compliance and optimization integrity | 10 | automatic |
| VQE-C7 Ideal-state accuracy at undisclosed geometries | 15 | automatic, formula in the notebook |
| VQE-C8 Noisy-hardware accuracy at undisclosed geometries | 10 | automatic, formula in the notebook |
| VQE-C9 Reported estimate and uncertainty | 5 | automatic |
| VQE-C10 Scientific interpretation, experiments and figures | 10 | manual |

The accuracy formulas for VQE-C7 and VQE-C8, the resource budget, and the penalties for budget violations are defined in Section 11 of the Challenge 3 notebook. They will not change after the event starts.

### Manual review form

Judges score the manual criteria with the following items:

| Item | Points |
|:--|--:|
| E91: Bell violation versus QBER versus security proof (Q1) | 3 |
| E91: derivation and quantitative comparison (Q2) | 3 |
| E91: statistical estimate (Q3) | 2 |
| E91: experiments, figures and stretch reasoning (Q4, Section 9) | 2 |
| Grover: reproducible sweeps and labelled figures (Section 9) | 3 |
| Grover: rotation picture and model of the noisy optimum (Q1, Q2) | 3 |
| Grover: resource table and statistically sound comparison (Q3) | 2 |
| Grover: implications without overclaiming (Q4) | 2 |
| VQE: active space, mapping, tapering, variational bound (Q1) | 2 |
| VQE: ansatz design and comparison (Q2) | 3 |
| VQE: optimizer, budget accounting, uncertainty (Q3) | 2 |
| VQE: dissociation curve, error decomposition, discussion (Q4, Section 9) | 3 |

### Ranking and tie-breakers

Participants are ranked by total points. Ties are broken, in order, by: (1) the higher Challenge 3 score; (2) the lower mean ideal-state error at the undisclosed geometries; (3) fewer two-qubit gates in the returned VQE circuits; (4) fewer total shots at the undisclosed geometries; (5) the higher Challenge 2 score. Participants who are still tied share the rank.

## 7. Competition rules

### Integrity of the provided code

| Rule | Situation | Consequence |
|:--|:--|:--|
| R1a | Formatting-only change to a protected cell (whitespace or comments) | Recorded; no penalty |
| R1b | Substantive change to protected code | Challenge score withheld for judge review; default outcome zero points for the challenge; deliberate circumvention of the grading leads to disqualification |
| R2 | Protected cell deleted | Challenge score withheld for judge review; default outcome zero points for the challenge |
| R3 | Protected cell duplicated | Identical copy: no penalty. Modified copy: treated as R1b |
| R4 | Protected cells reordered | Challenge score withheld for judge review; default outcome zero points for the challenge |
| R5 | Required function name or parameter list changed | Zero points for every criterion that uses that function |
| R6 | Code that tampers with provided objects or bypasses the resource budget | Manual review; the judges decide under these rules |
| R7 | Notebook missing | Zero points for the challenge |
| R8 | Notebook cannot be opened as a Jupyter notebook | Zero points for the challenge |
| R9 | Notebook misnamed | Graded with a warning if the challenge can be identified unambiguously; otherwise zero points |

Findings are never applied silently: every case is documented with evidence, and participants can appeal through the organizers before the final ranking is published.

### Fair play and academic integrity

* Submit your own work. Discuss ideas with other participants if you like, but do not share code.
* Cite external sources (papers, tutorials, documentation) in your notebook when you use them.
* **AI tools:** you may use AI assistants to help you understand concepts and theory, for example how the CHSH inequality or the Jordan-Wigner mapping works. The code you submit must be written by you; do not submit code produced by AI tools. Judges may ask you to explain your code.
* Do not hardcode results, read or guess hidden test data, or bypass the provided simulation helpers and budgets. Results must come from running your implementation.
* Do not attempt to access the network, other files or other submissions from your notebook during grading.

## 8. Submission checklist

1. Restart the kernel and run every notebook from top to bottom. Check that it finishes within the time limits.
2. Keep the original file names:
   * `01_e91_quantum_key_distribution.ipynb`
   * `02_grover_search_under_noise.ipynb`
   * `03_vqe_molecular_energy.ipynb`
3. Put the three notebooks in one folder named after you (letters, digits, `-` and `_` only), for example `abdullah_al_omar_galib/`. Notebooks you did not attempt can be left out; they score zero.
4. Compress the folder into a ZIP file using the same name, for example `abdullah_al_omar_galib.zip`.
5. Create your individual project on the [Devpost hackathon page](https://ibm-qiskit-hackathon.devpost.com/) using your name, and upload the ZIP file in the submission form before the deadline.

## 9. Package contents

```text
qiskit-quantum-hackathon/
  README.md                                   this guide
  requirements.txt                            pinned Python packages
  environment.yml                             conda environment
  notebooks/
    01_e91_quantum_key_distribution.ipynb
    02_grover_search_under_noise.ipynb
    03_vqe_molecular_energy.ipynb
  src/public_helpers.py                       provided helper functions
  data/lih_sto3g_active_space_public.json     LiH integrals for Challenge 3
  references/scientific_references.md         reading list
```

Good luck, and enjoy the problems.
