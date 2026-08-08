"""Build and smoke-test the distributable Faceledger CLI."""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "faceledger"


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    """Run one qualification command and return its standard output."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        rendered_command = " ".join(command)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {rendered_command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def _assert_distribution_contents(source_archive: Path, wheel: Path) -> None:
    """Verify both artifacts contain the import package and wheel launcher."""

    with tarfile.open(source_archive, "r:gz") as archive:
        source_names = set(archive.getnames())
    source_root = source_archive.name.removesuffix(".tar.gz")
    required_source_names = {
        f"{source_root}/faceledger/__init__.py",
        f"{source_root}/faceledger/__main__.py",
        f"{source_root}/faceledger/cli.py",
        f"{source_root}/pyproject.toml",
    }
    missing_source_names = required_source_names - source_names
    if missing_source_names:
        raise RuntimeError(
            f"Source distribution is missing: {sorted(missing_source_names)}"
        )
    bundled_source_weights = sorted(
        name for name in source_names if name.casefold().endswith(".h5")
    )
    if bundled_source_weights:
        raise RuntimeError(
            f"Source distribution bundles model weights: {bundled_source_weights}"
        )

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    required_wheel_suffixes = {
        "faceledger/__init__.py",
        "faceledger/__main__.py",
        "faceledger/cli.py",
        ".dist-info/entry_points.txt",
    }
    for suffix in required_wheel_suffixes:
        if not any(name.endswith(suffix) for name in wheel_names):
            raise RuntimeError(f"Wheel is missing an entry ending in {suffix}")
    bundled_wheel_weights = sorted(
        name for name in wheel_names if name.casefold().endswith(".h5")
    )
    if bundled_wheel_weights:
        raise RuntimeError(f"Wheel bundles model weights: {bundled_wheel_weights}")


def _assert_installed_grammar(
    prefix: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    """Smoke-test the complete public grammar through one installed route."""

    cases = (
        (("--help",), ("compare", "cache", "--version")),
        (
            ("compare", "--help"),
            (
                "source",
                "target_root",
                "--model {facenet512,arcface}",
                "--threshold",
                "--no-cache",
                "--no-recursive",
                "--result-file",
                "--log-file",
                "--no-progress",
            ),
        ),
        (("cache", "--help"), ("build", "rebuild", "trash")),
        (
            ("cache", "build", "--help"),
            ("maintenance_root", "--model", "--recursive", "--no-progress"),
        ),
        (
            ("cache", "rebuild", "--help"),
            ("maintenance_root", "--model", "--recursive", "--no-progress"),
        ),
        (
            ("cache", "trash", "--help"),
            ("maintenance_root", "--model", "--recursive", "--no-progress"),
        ),
    )
    for arguments, expected_fragments in cases:
        output = _run(
            (*prefix, *arguments),
            cwd=cwd,
            environment=environment,
        )
        for fragment in expected_fragments:
            if fragment not in output:
                rendered_command = " ".join((*prefix, *arguments))
                raise RuntimeError(f"{rendered_command} output is missing {fragment!r}")


def main() -> None:
    """Qualify built artifacts in an isolated locked installation."""

    package_version = version(PACKAGE_NAME)
    pinned_python = (PROJECT_ROOT / ".python-version").read_text().strip()
    clean_environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        clean_environment.pop(variable, None)

    with tempfile.TemporaryDirectory(prefix="faceledger-distribution-") as raw_path:
        workspace = Path(raw_path)
        distribution_directory = workspace / "dist"
        requirements_path = workspace / "runtime-requirements.txt"
        environment_path = workspace / "venv"
        smoke_path = workspace / "smoke"
        smoke_path.mkdir()

        _run(
            (
                "uv",
                "build",
                "--no-progress",
                "--out-dir",
                str(distribution_directory),
            ),
            cwd=PROJECT_ROOT,
            environment=clean_environment,
        )
        source_archive = (
            distribution_directory / f"{PACKAGE_NAME}-{package_version}.tar.gz"
        )
        wheel = (
            distribution_directory
            / f"{PACKAGE_NAME}-{package_version}-py3-none-any.whl"
        )
        if not source_archive.is_file() or not wheel.is_file():
            raise RuntimeError(
                f"Expected {source_archive.name} and {wheel.name} to be built"
            )
        _assert_distribution_contents(source_archive, wheel)

        _run(
            (
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements_path),
            ),
            cwd=PROJECT_ROOT,
            environment=clean_environment,
        )
        _run(
            ("uv", "venv", "--python", pinned_python, str(environment_path)),
            cwd=workspace,
            environment=clean_environment,
        )
        python = environment_path / "bin" / "python"
        launcher = environment_path / "bin" / "faceledger"
        _run(
            (
                "uv",
                "pip",
                "install",
                "--quiet",
                "--strict",
                "--python",
                str(python),
                "--requirements",
                str(requirements_path),
                str(wheel),
            ),
            cwd=workspace,
            environment=clean_environment,
        )

        python_version = _run(
            (str(python), "--version"),
            cwd=smoke_path,
            environment=clean_environment,
        ).strip()
        if python_version != f"Python {pinned_python}":
            raise RuntimeError(
                f"Expected Python {pinned_python}, installed {python_version}"
            )
        expected_version = f"{PACKAGE_NAME} {package_version}\n"
        launcher_version = _run(
            (str(launcher), "--version"),
            cwd=smoke_path,
            environment=clean_environment,
        )
        module_version = _run(
            (str(python), "-m", PACKAGE_NAME, "--version"),
            cwd=smoke_path,
            environment=clean_environment,
        )
        if launcher_version != expected_version or module_version != expected_version:
            raise RuntimeError("Installed routes did not report the packaged version")

        imported_from = Path(
            _run(
                (
                    str(python),
                    "-c",
                    "import faceledger; print(faceledger.__file__)",
                ),
                cwd=smoke_path,
                environment=clean_environment,
            ).strip()
        )
        if not imported_from.is_relative_to(environment_path):
            raise RuntimeError(f"Faceledger imported outside the venv: {imported_from}")

        _assert_installed_grammar(
            (str(launcher),),
            cwd=smoke_path,
            environment=clean_environment,
        )
        _assert_installed_grammar(
            (str(python), "-m", PACKAGE_NAME),
            cwd=smoke_path,
            environment=clean_environment,
        )

    print(
        f"Qualified {source_archive.name} and {wheel.name} with Python {pinned_python}."
    )


if __name__ == "__main__":
    main()
