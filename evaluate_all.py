"""Evaluate all three hackathon notebooks and estimate the total score (out of 300).

Usage (from the qiskit-quantum-hackathon folder, environment activated, with
test_e91.py, test_grover.py and test_vqe.py next to this file):

    python evaluate_all.py                         # tests + integrity + clean re-run
    python evaluate_all.py --original-zip ~/Downloads/qiskit-quantum-hackathon-participant-1.0.0.zip
    python evaluate_all.py --full-vqe              # 4 VQE runs like the grader (slower)
    python evaluate_all.py --skip-rerun            # skip the fresh-kernel re-execution

What it does for each notebook:
  1. Automatic tasks (90 pts): runs the matching test_*.py and reads its estimate.
  2. Integrity: compares every "# Provided Code: Do Not Modify" cell and every
     participant function signature with the original package (if --original-zip).
  3. Clean re-run: executes the notebook in a fresh kernel with a new QHACK_SEED,
     checks for errors and the time limit (15 / 20 / 30 min), counts figures.
  4. Manual part (10 pts): rough readiness check of experiments, figures and
     written answers. Judges score this; the number shown is only a checklist.
It never modifies your notebooks (re-runs are written to a temp folder).
Keep this file and the test files OUT of your submission folder.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="notebooks", help="folder with your three notebooks")
ap.add_argument("--original-zip", default=None, help="original participant package (for integrity check)")
ap.add_argument("--full-vqe", action="store_true")
ap.add_argument("--skip-rerun", action="store_true")
ap.add_argument("--seed", default="31337", help="QHACK_SEED for the clean re-run")
ARGS = ap.parse_args()

HERE = Path(__file__).resolve().parent
NBDIR = Path(ARGS.dir).resolve()
CHALLENGES = [
    ("Challenge 1 (E91)", "01_e91_quantum_key_distribution.ipynb", "test_e91.py", 15, "E91"),
    ("Challenge 2 (Grover)", "02_grover_search_under_noise.ipynb", "test_grover.py", 20, "GRV"),
    ("Challenge 3 (VQE)", "03_vqe_molecular_energy.ipynb", "test_vqe.py", 30, "VQE"),
]


def cells(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["cells"]


def src(cell):
    return "".join(cell["source"])


def first_line(cell):
    s = src(cell).strip()
    return s.splitlines()[0] if s else ""


# ------------------------------------------------------------- 1. automatic tests
def run_tests(script, nb_path):
    cmd = [sys.executable, str(HERE / script)]
    if script == "test_vqe.py":
        cmd += ["--nb", str(nb_path)] + (["--full"] if ARGS.full_vqe else [])
    else:
        cmd += [str(nb_path)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    out = p.stdout + p.stderr
    m = re.search(r"Estimated automatic score:\s*([\d.]+)/90", out)
    score = float(m.group(1)) if m else 0.0
    lines = [l for l in out.splitlines() if l.startswith(("FAIL", "ERROR", "TODO"))]
    return score, lines, time.time() - t0, out if not m else ""


# ------------------------------------------------------------- 2. integrity
def signatures(cell_src):
    sigs = {}
    try:
        tree = ast.parse(cell_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            defaults = [ast.unparse(d) for d in a.defaults + [d for d in a.kw_defaults if d]]
            sigs[node.name] = (tuple(names), tuple(defaults), ast.unparse(a))
    return sigs


def normalize(code):
    return "\n".join(l.split("#")[0].rstrip() for l in code.strip().splitlines() if l.split("#")[0].strip())


def integrity(mine, original):
    issues = []
    mc, oc = cells(mine), cells(original)
    prov_m = [src(c) for c in mc if c["cell_type"] == "code" and first_line(c).startswith("# Provided Code")]
    prov_o = [src(c) for c in oc if c["cell_type"] == "code" and first_line(c).startswith("# Provided Code")]
    if len(prov_m) != len(prov_o):
        issues.append(f"{len(prov_m)} protected cells, original has {len(prov_o)} (deleted/duplicated?)")
    for i, (a, b) in enumerate(zip(prov_m, prov_o)):
        if normalize(a) != normalize(b):
            issues.append(f"protected cell #{i+1} was modified or reordered")
    part_o = {first_line(c): signatures(src(c)) for c in oc
              if c["cell_type"] == "code" and first_line(c).startswith("# Participant Code: Task")}
    part_m = {first_line(c): signatures(src(c)) for c in mc
              if c["cell_type"] == "code" and first_line(c).startswith("# Participant Code: Task")}
    for tag, sig in part_o.items():
        if tag not in part_m:
            issues.append(f"participant cell missing: {tag}")
            continue
        if part_m[tag] is None:
            issues.append(f"syntax error in {tag}")
            continue
        for fname, args in (sig or {}).items():
            if fname not in part_m[tag]:
                issues.append(f"{fname} missing in {tag}")
            elif part_m[tag][fname][:2] != args[:2]:
                issues.append(f"signature of {fname} changed: ({part_m[tag][fname][2]}) vs original ({args[2]})")
            elif part_m[tag][fname][2] != args[2]:
                issues.append(f"note: type hints of {fname} differ from the original (harmless, but keep them to be safe)")
    return issues


# ------------------------------------------------------------- 3. clean re-run
def rerun(nb_path, limit_min):
    tmp = Path(tempfile.mkdtemp())
    out_nb = tmp / nb_path.name
    env = dict(os.environ, QHACK_SEED=ARGS.seed)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--allow-errors", f"--ExecutePreprocessor.timeout={limit_min*60}",
                        "--output", str(out_nb), str(nb_path)],
                       capture_output=True, text=True, env=env, cwd=nb_path.parent)
    dt = time.time() - t0
    if p.returncode != 0 and not out_nb.exists():
        return dict(ok=False, minutes=dt / 60, figures=0, failed_checks=[],
                    errors=["could not execute (is jupyter installed? pip install jupyter): " + p.stderr[-300:]])
    if not out_nb.exists():
        return dict(ok=False, minutes=dt / 60, errors=[p.stderr[-400:]], figures=0, failed_checks=[])
    errors, figures, failed = [], 0, []
    for c in cells(out_nb):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append(f"{first_line(c)[:50]}: {o.get('ename')}: {o.get('evalue','')[:80]}")
            if "image/png" in o.get("data", {}) or "image/svg+xml" in o.get("data", {}):
                figures += 1
            text = "".join(o.get("text", "")) if isinstance(o.get("text", ""), list) else o.get("text", "")
            for line in str(text).splitlines():
                if "FAILED" in line or "ERROR" in line or "not implemented" in line:
                    failed.append(line.strip()[:100])
    return dict(ok=not errors and dt / 60 <= limit_min, minutes=dt / 60, errors=errors,
                figures=figures, failed_checks=failed)


# ------------------------------------------------------------- 4. manual readiness
def manual_readiness(nb_path, tag, figures):
    cs = cells(nb_path)
    answers = [c for c in cs if c["cell_type"] == "markdown" and src(c).lstrip().startswith("**Your answer")]
    filled = 0
    for c in answers:
        body = re.sub(r"\*\*Your answer.*?\*\*", "", src(c))
        body = re.sub(r"^_.*?_\s*$", "", body.strip(), flags=re.S | re.M)  # drop the italic prompt
        if len(body.strip()) > 150:
            filled += 1
    x = [c for c in cs if c["cell_type"] == "code" and first_line(c).startswith(f"# Participant Code: Task {tag}-X")]
    exp_code = sum(len(normalize(src(c)).splitlines()) for c in x)
    extra_cells = sum(1 for c in cs if c["cell_type"] == "code" and not first_line(c).startswith("#"))
    pts = 0.0
    pts += 4.0 * filled / max(1, len(answers))
    pts += 3.0 if exp_code + extra_cells > 10 else (1.5 if exp_code + extra_cells > 0 else 0.0)
    pts += 3.0 if figures >= 3 else figures
    return pts, f"answers {filled}/{len(answers)}, experiment code lines {exp_code} (+{extra_cells} extra cells), figures {figures}"


# ------------------------------------------------------------- main
def main():
    orig_dir = None
    if ARGS.original_zip:
        orig_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(ARGS.original_zip) as z:
            z.extractall(orig_dir)
    grand, report = 0.0, []
    for title, fname, script, limit, tag in CHALLENGES:
        nb = NBDIR / fname
        print(f"\n=== {title}: {fname} ===")
        if not nb.exists():
            print("  not found -> 0 points (left out of the submission)")
            report.append((title, 0, 0, 0))
            continue
        auto, problems, secs, raw = run_tests(script, nb)
        print(f"  automatic tests: {auto:.1f}/90 ({secs:.0f}s)")
        for l in problems:
            print("    " + l[:160])
        if raw:
            print("    test script crashed:\n" + raw[-800:])

        if orig_dir:
            originals = list(orig_dir.rglob(fname))
            iss = integrity(nb, originals[0]) if originals else ["original notebook not found in zip"]
            print("  integrity: " + ("OK" if all(i.startswith("note:") for i in iss) else "PROBLEMS"))
            for i in iss:
                print("    ! " + i)
            if any(not i.startswith("note:") for i in iss):
                print("    -> substantive changes to protected code default to 0 for this challenge")
        else:
            print("  integrity: skipped (pass --original-zip to compare with the package)")

        figures = 0
        if not ARGS.skip_rerun:
            r = rerun(nb, limit)
            figures = r["figures"]
            status = "OK" if r["ok"] else "PROBLEM"
            print(f"  clean re-run (QHACK_SEED={ARGS.seed}): {status}, {r['minutes']:.1f}/{limit} min, "
                  f"{figures} figures")
            for e in r["errors"][:8]:
                print("    error: " + e)
            for f in sorted(set(r["failed_checks"]))[:8]:
                print("    check: " + f)
            if r["minutes"] > limit:
                print("    -> over the time limit")
        manual, info = manual_readiness(nb, tag, figures)
        print(f"  manual part readiness: ~{manual:.1f}/10 ({info}) - judges decide this score")
        total = auto + manual
        grand += total
        report.append((title, auto, manual, total))

    print("\n================ SUMMARY ================")
    for title, auto, manual, total in report:
        print(f"  {title:<22} auto {auto:5.1f}/90  manual~{manual:4.1f}/10  total~{total:5.1f}/100")
    print(f"  {'TOTAL':<22} ~{grand:.1f}/300")
    print("Estimates only: hidden tests use other keys, angles, seeds and bond lengths.")
    print("Tie-breakers: Challenge 3 score, mean VQE ideal error, fewer CX, fewer VQE shots, Challenge 2 score.")


if __name__ == "__main__":
    main()
