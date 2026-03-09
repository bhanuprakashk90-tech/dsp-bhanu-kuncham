#!/usr/bin/env python3
"""Unified validation script for DSP practical work submissions.

Validates student submissions for PW0, PW1, and PW2 to ensure they meet
the requirements for automatic grading.

Usage:
    python validate_submission.py --pw0
    python validate_submission.py --pw1
    python validate_submission.py --pw2
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

# Constants
PW0_FILES = ["notebooks/my-1st-notebook.ipynb", "requirements.txt"]
PW1_FILES = ["notebooks/house-prices-modeling.ipynb", "requirements.txt", ".gitignore"]
PW2_FILES = [
    "house_prices/__init__.py",
    "house_prices/preprocess.py",
    "house_prices/train.py",
    "house_prices/inference.py",
    "requirements.txt",
    "notebooks/model-industrialization.ipynb",
    "notebooks/model-industrialization-final.ipynb",
]
PW2_ALLOWED_PACKAGES = [
    "pandas", "numpy", "sklearn", "scikit-learn", "matplotlib", "seaborn",
    "joblib", "flake8", "openpyxl", "pyarrow", "pathlib", "os", "sys",
    "typing", "house_prices", "json", "warnings", "re",
]
PW2_SIGNATURES = {
    "preprocess": (["df", "is_training"], "house_prices/preprocess.py"),
    "build_model": (["filepath"], "house_prices/train.py"),
    "make_predictions": (["filepath"], "house_prices/inference.py"),
}


class ValidationContext:
    """Tracks validation state and issues."""

    def __init__(self, pw_name: str):
        self.pw_name = pw_name
        self.issues: list[str] = []
        self.check_num = 0

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)

    def next_check(self, name: str) -> None:
        self.check_num += 1
        print(f"\n[{self.check_num}] {name}...")

    def ok(self, msg: str) -> None:
        print(f"  ✅ {msg}")

    def fail(self, msg: str) -> None:
        self.add_issue(msg)
        print(f"  ❌ {msg}")

    def skip(self, msg: str) -> None:
        print(f"  ⏭️  {msg}")


# Common utilities
def run_git(args: list[str]) -> tuple[bool, str]:
    """Run git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def check_on_main_branch(ctx: ValidationContext, dev: bool = False) -> None:
    if dev:
        ctx.next_check("Checking git branch (skipped in dev mode)")
        ctx.skip("Dev mode - branch check skipped. Remember to merge to main before final submission!")
        return
    ctx.next_check("Checking git branch")
    success, branch = run_git(["branch", "--show-current"])
    if not success:
        ctx.fail("Git error - cannot determine branch")
    elif branch != "main":
        ctx.fail(f"On branch '{branch}', expected 'main'")
    else:
        ctx.ok("On main branch")


def check_files_exist(ctx: ValidationContext, files: list[str]) -> None:
    ctx.next_check("Checking required files")
    for f in files:
        if Path(f).exists():
            ctx.ok(f)
        else:
            ctx.fail(f"Missing: {f} - create this file")


def check_folder_exists(ctx: ValidationContext, folder: str) -> None:
    if Path(folder).is_dir():
        ctx.ok(f"{folder}/")
    else:
        ctx.fail(f"Missing folder: {folder}/ - create with: mkdir {folder}")


def check_gitignore_contains(ctx: ValidationContext, pattern: str) -> None:
    ctx.next_check(f"Checking .gitignore contains '{pattern}'")
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        ctx.skip(".gitignore missing")
        return
    if pattern in gitignore.read_text():
        ctx.ok(f".gitignore contains '{pattern}'")
    else:
        ctx.fail(f".gitignore should contain '{pattern}'")


def check_notebook_has_outputs(ctx: ValidationContext, path: str) -> bool:
    ctx.next_check(f"Checking notebook outputs: {path}")
    nb_path = Path(path)
    if not nb_path.exists():
        ctx.skip("Notebook missing")
        return False
    try:
        nb = json.loads(nb_path.read_text())
        cells_with_output = sum(
            1 for c in nb.get("cells", [])
            if c.get("cell_type") == "code" and c.get("outputs")
        )
        if cells_with_output > 0:
            ctx.ok(f"Notebook has outputs ({cells_with_output} cells)")
            return True
        else:
            ctx.fail("Notebook has no cell outputs - run all cells (Cell > Run All) and save before submitting")
            return False
    except (json.JSONDecodeError, KeyError) as e:
        ctx.fail(f"Cannot read notebook: {e}")
        return False


def read_text_safe(path: Path) -> str:
    """Read text file with fallback encodings."""
    for enc in ["utf-8", "utf-8-sig", "utf-16", "latin-1"]:
        try:
            return path.read_bytes().decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def check_requirements_contains(ctx: ValidationContext, package: str) -> None:
    ctx.next_check(f"Checking requirements.txt contains '{package}'")
    req_path = Path("requirements.txt")
    if not req_path.exists():
        ctx.skip("requirements.txt missing")
        return
    content = read_text_safe(req_path).lower()
    if package.lower() in content:
        ctx.ok(f"requirements.txt contains '{package}'")
    else:
        ctx.fail(f"requirements.txt should contain '{package}'")


def get_notebook_outputs_text(path: str) -> str:
    """Extract all text outputs from a notebook."""
    try:
        nb = json.loads(Path(path).read_text())
        texts = []
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                if "text" in output:
                    texts.extend(output["text"])
                if "data" in output and "text/plain" in output["data"]:
                    texts.extend(output["data"]["text/plain"])
        return "\n".join(texts)
    except Exception:
        return ""


def print_report(ctx: ValidationContext) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    if ctx.issues:
        print(f"❌ {len(ctx.issues)} issue(s) found:\n")
        for i, issue in enumerate(ctx.issues, 1):
            print(f"  {i}. {issue}")
        print("\nFix these issues before submitting!")
    else:
        print("✅ All checks passed!")

    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT")
    print("   This script only validates basic submission requirements.")
    print("   Passing all checks does NOT guarantee a good grade.")
    print()
    print("   You are responsible for:")
    print("   - Following ALL assignment instructions")
    print("   - Code quality and correctness")
    print("   - Proper implementation of grading criteria")
    print()
    print("   Review the assignment instructions before submitting.")
    print("=" * 60)


# PW0 validation
def check_branch_exists(ctx: ValidationContext, branch: str) -> None:
    ctx.next_check(f"Checking branch '{branch}' exists")
    success, output = run_git(["branch", "-a"])
    if not success:
        ctx.fail("Git error - cannot list branches")
        return
    # Exact match: branch name as whole word
    branches = [b.strip().replace("* ", "").split("/")[-1] for b in output.splitlines()]
    if branch in branches:
        ctx.ok(f"Branch '{branch}' exists")
    else:
        ctx.fail(f"Branch '{branch}' not found (should not be deleted)")


def check_branch_merged(ctx: ValidationContext, branch: str) -> None:
    ctx.next_check(f"Checking branch '{branch}' is merged to main")
    success, output = run_git(["branch", "--merged", "main"])
    if not success:
        ctx.skip("Cannot check merged branches")
        return
    branches = [b.strip().replace("* ", "") for b in output.splitlines()]
    if branch in branches:
        ctx.ok(f"Branch '{branch}' is merged to main")
    else:
        ctx.fail(f"Branch '{branch}' is not merged to main - run: git checkout main && git merge {branch}")


def check_notebook_has_array_output(ctx: ValidationContext, path: str) -> None:
    ctx.next_check("Checking notebook has numpy array output")
    output_text = get_notebook_outputs_text(path)
    if not output_text:
        ctx.skip("No notebook output to check")
        return
    # Look for array-like output: brackets with numbers
    if re.search(r'\[[\s\d\.\-,\n\]]+\]', output_text):
        ctx.ok("Notebook contains array-like output")
    else:
        ctx.fail("Notebook should display a numpy array - add a cell with: print(your_array)")


def validate_pw0(ctx: ValidationContext, dev: bool = False) -> None:
    check_on_main_branch(ctx, dev)
    check_files_exist(ctx, ["requirements.txt", ".gitignore", "notebooks/my-1st-notebook.ipynb"])
    check_requirements_contains(ctx, "jupyter")  # pw0_06
    check_notebook_has_outputs(ctx, "notebooks/my-1st-notebook.ipynb")  # pw0_10
    check_branch_exists(ctx, "my-1st-jupyter-notebook")  # pw0_11
    check_branch_merged(ctx, "my-1st-jupyter-notebook")  # pw0_12


# PW1 validation
def validate_pw1(ctx: ValidationContext, dev: bool = False) -> None:
    check_on_main_branch(ctx, dev)
    check_files_exist(ctx, ["notebooks/house-prices-modeling.ipynb"])  # pw1_01
    check_gitignore_contains(ctx, "data")  # pw1_02
    check_notebook_has_outputs(ctx, "notebooks/house-prices-modeling.ipynb")  # pw1_05
    check_branch_exists(ctx, "pw1")  # pw1_14
    check_branch_merged(ctx, "pw1")  # pw1_15


# PW2 validation
def check_environment(ctx: ValidationContext) -> bool:
    ctx.next_check("Checking environment")
    missing = []
    for pkg in ["pandas", "numpy", "sklearn", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        ctx.fail(f"Cannot import: {', '.join(missing)}")
        print("     Please activate your virtual environment and try again.")
        return False
    ctx.ok("Required packages available")
    return True


def check_function_signatures(ctx: ValidationContext) -> None:
    ctx.next_check("Checking function signatures")
    for func_name, (expected_params, filepath) in PW2_SIGNATURES.items():
        if not Path(filepath).exists():
            ctx.skip(f"{func_name} - file missing")
            continue
        try:
            tree = ast.parse(Path(filepath).read_text())
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    params = [arg.arg for arg in node.args.args]
                    if params == expected_params:
                        ctx.ok(f"{func_name}({', '.join(params)})")
                    else:
                        ctx.fail(
                            f"{func_name}: wrong signature. "
                            f"Change 'def {func_name}({', '.join(params)})' "
                            f"to 'def {func_name}({', '.join(expected_params)})'"
                        )
                        if func_name in ("build_model", "make_predictions"):
                            param = expected_params[0]
                            print(f"     Note: {param} is a file path - "
                                  f"load CSV with: df = pd.read_csv({param})")
                    found = True
                    break
            if not found:
                ctx.fail(
                    f"{func_name} not found in {filepath}. "
                    f"Add: def {func_name}({', '.join(expected_params)}): ..."
                )
        except SyntaxError as e:
            ctx.fail(f"{func_name} - Syntax error: {e}")


def check_imports(ctx: ValidationContext) -> None:
    ctx.next_check("Checking imports")
    unauthorized = []
    for py_file in Path("house_prices").glob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        pkg = alias.name.split(".")[0]
                        if not pkg.startswith("_") and pkg not in PW2_ALLOWED_PACKAGES:
                            unauthorized.append(f"{pkg} (in {py_file.name})")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    pkg = node.module.split(".")[0]
                    if not pkg.startswith("_") and pkg not in PW2_ALLOWED_PACKAGES:
                        unauthorized.append(f"{pkg} (in {py_file.name})")
        except SyntaxError:
            pass
    if unauthorized:
        allowed = ("pandas, numpy, sklearn, joblib, matplotlib, seaborn, "
                   "os, sys, json, re, typing, warnings, pathlib")
        for pkg in set(unauthorized):
            ctx.fail(f"Unauthorized import: {pkg}. Allowed: {allowed}")
    else:
        ctx.ok("All imports allowed")


def check_flake8(ctx: ValidationContext) -> None:
    ctx.next_check("Running flake8")
    if not Path("house_prices").exists():
        ctx.skip("house_prices/ missing")
        return
    try:
        result = subprocess.run(
            ["flake8", "house_prices", "--count", "--select=E,W,F",
             "--max-line-length=120"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            lines = result.stdout.strip().split("\n")
            count = lines[-1] if lines else "?"
            ctx.fail(f"flake8: {count} error(s) found")
            for line in lines[:5]:
                if line and not line.isdigit():
                    print(f"     {line}")
            if len(lines) > 6:
                print("     ... and more. Run 'flake8 house_prices' to see all")
        else:
            ctx.ok("No flake8 errors")
    except FileNotFoundError:
        ctx.fail("flake8 not installed")
    except subprocess.TimeoutExpired:
        ctx.fail("flake8 timed out")


def check_data_files(ctx: ValidationContext) -> None:
    ctx.next_check("Checking data files (local only)")
    for f in ["data/train.csv", "data/test.csv"]:
        if Path(f).exists():
            ctx.ok(f)
        else:
            ctx.fail(f"Missing: {f} (required locally, do not push)")


def check_runtime(ctx: ValidationContext) -> None:
    ctx.next_check("Running build_model")
    if not Path("data/train.csv").exists():
        ctx.skip("data/train.csv missing")
        return
    if not Path("house_prices/train.py").exists():
        ctx.skip("house_prices/train.py missing")
        return

    # Clear models directory
    models_dir = Path("models")
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        models_dir.mkdir()

    try:
        sys.path.insert(0, str(Path.cwd()))
        from house_prices.train import build_model
        result = build_model("data/train.csv")
        if isinstance(result, dict):
            ctx.ok(f"build_model returns dict: {result}")
        else:
            ctx.fail(f"build_model should return dict with metrics "
                     f"(e.g., {{'rmse': 0.5}}), got {type(result).__name__}")
            return
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        for frame in reversed(tb):
            if "house_prices" in frame.filename:
                ctx.fail(f"build_model failed at {frame.filename}:{frame.lineno}: {e}")
                print(f"     --> {frame.line}")
                break
        else:
            ctx.fail(f"build_model failed: {e}")
        return

    # Check artifacts
    ctx.next_check("Checking artifacts")
    artifacts = list(models_dir.glob("*"))
    if artifacts:
        ctx.ok(f"Artifacts created: {len(artifacts)} file(s)")
    else:
        ctx.fail("No artifacts in models/ - use joblib.dump(model, 'models/model.joblib') in build_model")
        return

    # Run inference
    ctx.next_check("Running make_predictions")
    if not Path("data/test.csv").exists():
        ctx.skip("data/test.csv missing")
        return

    try:
        from house_prices.inference import make_predictions
        import numpy as np
        result = make_predictions("data/test.csv")
        if isinstance(result, np.ndarray):
            ctx.ok(f"make_predictions returns ndarray (shape: {result.shape})")
        else:
            ctx.fail(
                f"make_predictions should return ndarray, got {type(result).__name__}"
            )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        for frame in reversed(tb):
            if "house_prices" in frame.filename:
                ctx.fail(f"make_predictions failed at {frame.filename}:{frame.lineno}: {e}")
                print(f"     --> {frame.line}")
                break
        else:
            ctx.fail(f"make_predictions failed: {e}")


def validate_pw2(ctx: ValidationContext, dev: bool = False) -> None:
    if not check_environment(ctx):
        return
    check_on_main_branch(ctx, dev)
    check_files_exist(ctx, PW2_FILES)
    ctx.next_check("Checking models folder")
    check_folder_exists(ctx, "models")
    check_function_signatures(ctx)
    check_imports(ctx)
    check_flake8(ctx)
    check_data_files(ctx)
    check_runtime(ctx)
    check_notebook_has_outputs(ctx, "notebooks/model-industrialization-final.ipynb")
    check_branch_exists(ctx, "pw2")
    check_branch_merged(ctx, "pw2")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DSP practical work submissions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pw0", action="store_true", help="Validate PW0 submission")
    group.add_argument("--pw1", action="store_true", help="Validate PW1 submission")
    group.add_argument("--pw2", action="store_true", help="Validate PW2 submission")
    parser.add_argument(
        "--dev", action="store_true",
        help="Development mode - skip 'must be on main' check"
    )
    args = parser.parse_args()

    if args.pw0:
        pw_name = "PW0"
        validate_fn = validate_pw0
    elif args.pw1:
        pw_name = "PW1"
        validate_fn = validate_pw1
    else:
        pw_name = "PW2"
        validate_fn = validate_pw2

    print("=" * 60)
    print(f"{pw_name} Submission Validation")
    print("=" * 60)

    ctx = ValidationContext(pw_name)
    validate_fn(ctx, dev=args.dev)
    print_report(ctx)

    return 1 if ctx.issues else 0


if __name__ == "__main__":
    sys.exit(main())
