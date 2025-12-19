# ============================================================
# FIN41660 Crisis Forecaster — Submission ZIP Builder
# ------------------------------------------------------------
# Purpose:
#   Package the Crisis Forecaster project into a single ZIP file
#   for submission. The archive includes:
#     - src/ (model + utilities code)
#     - app/app.py (Streamlit dashboard)
#     - forecasting_script.py (standalone reproducible pipeline)
#     - requirements.txt, README.md
#     - tests/ (pytest suite)
#     - report/ outputs (figures, CSVs, summaries) if present
#     - data/*.csv (repo-safe derived inputs) if present
#
# Notes:
#   - Uses ONLY the Python standard library for portability.
#   - Intended to be run from the project root directory.
# ============================================================

"""
Helper utility to bundle the Crisis Forecaster project for FIN41660 submission.

Creates a ZIP archive with code, requirements, and report outputs using only the
standard library. Run from the project root: `python make_submission_zip.py`.
"""

# Future annotations allows using modern type hints on older Python versions
from __future__ import annotations

# Path provides OS-independent filesystem paths
from pathlib import Path

# zipfile is the standard library tool for writing ZIP archives
import zipfile


# Absolute path to the project root folder (directory containing this script)
BASE_DIR = Path(__file__).resolve().parent

# Output ZIP file path (will be created/overwritten in the project root)
TARGET_ZIP = BASE_DIR / "Crisis_Forecaster_Submission.zip"


def _require_dirs(dirs: list[Path]) -> None:
    """
    Ensure required directories exist before creating the submission ZIP.

    Parameters
    ----------
    dirs : list[Path]
        List of directories that must exist.

    Raises
    ------
    FileNotFoundError
        If any required directory is missing.
    """
    # Collect any missing required folders
    missing = [d for d in dirs if not d.exists()]

    # If something is missing, raise a clear error listing what was not found
    if missing:
        missing_str = ", ".join(str(d) for d in missing)
        raise FileNotFoundError(f"Required directories missing: {missing_str}")


def collect_files() -> list[Path]:
    """
    Gather project assets to include in the submission archive.

    Returns
    -------
    list[Path]
        Sorted list of file paths to be written into the ZIP.
    """
    # Key project locations
    src_dir = BASE_DIR / "src"
    app_file = BASE_DIR / "app" / "app.py"
    report_dir = BASE_DIR / "report"
    tests_dir = BASE_DIR / "tests"
    data_dir = BASE_DIR / "data"

    # Check that core directories exist (these should always be present for submission)
    _require_dirs([src_dir, app_file.parent, report_dir, tests_dir])

    # Use a set to avoid duplicates
    files: set[Path] = set()

    # Include all Python source code under src/
    files.update(src_dir.rglob("*.py"))

    # Include Streamlit entrypoint
    files.add(app_file)

    # Include standalone pipeline script
    files.add(BASE_DIR / "forecasting_script.py")

    # Include dependency list and project documentation
    files.add(BASE_DIR / "requirements.txt")
    files.add(BASE_DIR / "README.md")

    # Include all test files
    files.update(tests_dir.rglob("*.py"))

    # If report/ exists, include everything in it (CSVs, PNGs, summaries, etc.)
    if report_dir.exists():
        files.update(report_dir.rglob("*"))

    # If data/ exists, include only CSVs (keeps submission small and consistent)
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            files.add(csv_file)

    # Return only files (not directories) and sort for reproducibility
    return sorted(f for f in files if f.is_file())


def create_zip(files: list[Path]) -> Path:
    """
    Write the collected files into the submission ZIP.

    Parameters
    ----------
    files : list[Path]
        The file paths that should be written into the ZIP.

    Returns
    -------
    Path
        The path to the created ZIP file.
    """
    # Open the ZIP in write mode (overwrites if it already exists)
    # ZIP_DEFLATED compresses the contents to reduce file size
    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            # Store files with a relative path inside the ZIP (keeps structure clean)
            arcname = file_path.relative_to(BASE_DIR)
            zf.write(file_path, arcname)

    # Return the output ZIP path for printing / downstream use
    return TARGET_ZIP


def main() -> None:
    """
    Main entry point:
      1) Collect files
      2) Create ZIP
      3) Print confirmation path
    """
    files = collect_files()
    zip_path = create_zip(files)
    print(f"Created submission archive at {zip_path}")


# Only run when executed directly (not when imported)
if __name__ == "__main__":
    main()

# comments by chatgpt edited by group ....
