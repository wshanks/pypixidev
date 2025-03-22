"""Tool to generate a pixi dev environment for a non-pixi Python project"""

# TODO: Do something with extras and markers like platform specific dependencies?

# TODO: Clean out Python specific versions / handle stuff like .post1

# TODO: conda.api is used to query the conda package index for package names to
# decide if they are available or should be left as Python dependencies.
# conda.api is annoying because conda can not be pip-installed, so this script
# can only be run within a conda environment (only supported installation
# method of conda). Is there a suitable alternative? grayskull hits
# anaconda.org with a separate request for each package which seems slow and
# wasteful (but gives the most up to date information). `pixi search` is
# another option but requires a subprocess per package which is a bit slow even
# though pixi is generally fast because loading the 10's of MB's of repo data
# each time takes time versus loading it once and checking all the packages in
# a single process. Loading and checking the repodata directly is an option but
# that requires directly managing refreshing the cache and relevant channels.
# That is the part using conda.api is giving us for now.

import configparser
import itertools
import json
import re
import shutil
import tempfile
import tomllib
from argparse import ArgumentParser
from collections import deque
from pathlib import Path
from subprocess import run
from typing import Any, Deque

import conda.api
import pkginfo
from grayskull.strategy.pypi import PYPI_CONFIG
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from ruamel.yaml import YAML


# pixi.toml allows relative paths but pixi add does not so we need to use this
# to make the path absolute and then hack it out
EDITABLE_PLACEHOLDER_PATH = "/magicplaceholderstringxxx/"
INLINE_COMMENT = re.compile(r"(\s+#.*$)")


def read_simple_requirements(path: Path) -> list[str]:
    """Read requirement specifiers from a simple requirements file

    Note that this is similar what ``tool.setuptools.dynamic.dependencies``
    supports, which is just lines with PEP508 specifies, comments, and
    whitespace and not lines starting with ``-r``, ``-e``, etc.

    Args:
        path: path to the file

    Returns:
        The list of requirements from the file
    """
    return [
        INLINE_COMMENT.split(line)[0].strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def check_pyproject(path: Path) -> bool:
    """Check if pyproject.toml file specifies dependencies"""
    if not path.exists():
        return False
    config = tomllib.loads(path.read_text())
    project = config.get("project", {})
    return bool(project.get("dependencies")) or "dependencies" in project.get(
        "dynamic", {}
    )


def gather_pyproject_reqs(pyprojects: list[str]) -> tuple[list[str], str]:
    """Gather all requirements specifiers from pyproject.toml files

    Args:
        pyprojects: list of paths to pyproject.toml files, optionally with
            trailing comma separated lists of extras enclosed in square
            brackets (like "./pyproject.toml[dev,docs]")

    Returns:
        tuple with list of Python dependency specifiers and set of version
        specifiers for python-requires as a string.
    """
    requirements = []
    requires_python = []

    for pyproject in pyprojects:
        name, extras = _split_extras(pyproject)

        config = tomllib.loads(Path(name).read_text())

        project = config.get("project", {})
        if "dependencies" in project:
            requirements.extend(project["dependencies"])
        elif "dependencies" in project.get("dynamic", {}) and (
            setuptools_dyn := _nested_get(
                config, "tool.setuptools.dynamic.dependencies.file"
            )
        ):
            for file in setuptools_dyn:
                requirements.extend(read_simple_requirements(Path(name).parent / file))
        else:
            ValueError(f"Unable to process {pyproject}")

        for extra in extras:
            if (
                "optional-dependencies" in project
                and extra in pyproject["optional-dependencies"]
            ):
                requirements.extend(project["optional-dependencies"][extra])
            elif "optional-dependencies" in project.get("dynamic", {}) and (
                setuptools_dyn := _nested_get(
                    config,
                    f"tool.setuptools.dynamic.optional-dependencies.{extra}.file",
                )
            ):
                for file in setuptools_dyn:
                    requirements.extend(
                        read_simple_requirements(Path(name).parent / file)
                    )
            else:
                ValueError(f"Unable to process extra {extra} for {pyproject}")

        if "requires-python" in project:
            requires_python.append(project["requires-python"])

    return requirements, ",".join(requires_python)


def gather_setuptools_reqs(setuptools_paths: list[str]) -> tuple[list[str], str]:
    """Gather all requirements specifiers from setup.cfg files

    Args:
        setuptools_paths: list of paths to setup.cfg files, optionally with
            trailing comma separated lists of extras enclosed in square
            brackets (like "./setup.cfg[dev,docs]")

    Returns:
        tuple with list of Python dependency specifiers and set of version
        specifiers for python-requires as a string.
    """
    requirements = []
    requires_python = []

    for setup in setuptools_paths:
        name, extras = _split_extras(setup)

        config = configparser.ConfigParser()
        config.read(name)
        if "options" not in config or "install_requires" not in config["options"]:
            raise ValueError(f"No install_requires found in {name}")
        install_requires = config["options"]["install_requires"].strip()
        if install_requires.strip().startswith("file:"):
            requirements.extend(
                read_simple_requirements(
                    Path(name).parent / install_requires[len("file:") :]
                )
            )
        else:
            requirements.extend(r for r in install_requires.splitlines() if r)

        for extra in extras:
            if "options.extras_require" in config and extra in config["options.extras_require"]:
                extras_require = config["options.extras_require"][extra]
            else:
                raise ValueError(f"Could not process extra {extra} for {setup}")
            if extras_require.startswith("file:"):
                requirements.extend(
                    read_simple_requirements(
                        Path(name).parent / extras_require[len("file:") :].strip()
                    )
                )
            else:
                requirements.extend(r for r in extras_require.splitlines() if r)

        if "options" in config and "python_requires" in config["options"]:
            requires_python.append(config["options"]["python_requires"])

    return requirements, ",".join(requires_python)


def gatther_requirements_reqs(requirements_in: list[str]) -> list[str]:
    """Gather requirement specifiers from requirements files"""
    requirements = []

    loaded: set[Path] = set()

    def _extend_specs(req_path: Path, specs: Deque[str], loaded: set[Path]):
        """Extend specs with new file's content adjusting context for relative -r entries"""
        new_specs = read_simple_requirements(req_path)
        for spec in new_specs:
            if spec.startswith("-r"):
                new_file = spec[len("-r") :].strip()
                specs.append(f"-r{req_path.parent / new_file}")
            else:
                specs.append(spec)

    for req_file in requirements_in:
        path = Path(req_file).resolve()
        loaded.add(path)
        parent = path.parent
        specs = deque(read_simple_requirements(path))
        while specs:
            spec = specs.popleft()
            if spec.startswith("-r"):
                new_path = (parent / spec[len("-r") :].strip()).resolve()
                if new_path not in loaded:
                    _extend_specs(new_path, specs, loaded)
                    loaded.add(new_path)
            elif spec.startswith("-"):
                raise ValueError(f"Unsupported requirements file line: {spec}")
            else:
                requirements.append(spec)

    return requirements


def _nested_get(table: dict[str, Any], path: str) -> Any:
    result = table
    for key in path.split("."):
        if key not in result:
            raise KeyError(f"{path} not found in {table}")
        result = result[key]

    return result


def _split_extras(requirement: str) -> tuple[str, list[str]]:
    if extras_match := re.match(r"(?P<path>.*)\[(?P<extras>.*)\]$", requirement):
        name = extras_match.group("path")
        extras_spec = extras_match.group("extras") or ""
    else:
        name = requirement
        extras_spec = ""

    if extras_spec:
        extras = extras_spec.split(",")
    else:
        extras = []

    return name, extras


def copy_editables_to_packages(editables: list[str], packages: list[str]):
    """Copy editable paths into packages with extras"""
    for editable in editables:
        if " @ " in editable:
            name_spec, _, path = editable.partition(" @ ")
            name, extras = _split_extras(name_spec)
            extras_str = f"[{','.join(extras)}]" if extras else ""
            packages.append(f"{path}{extras_str}")
        else:
            packages.append(editable)


def detect_packages(
    packages: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Detect how packages' dependencies are specified

    Args:
        packages: list of file paths to local packages to check

    Returns:
        tuple with:

            + list of paths with dependencies in pyproject.toml files
            + list of paths with dependencies in setup.cfg files
            + list of paths with requirements.txt files and no pyproject.toml
              or setup.cfg file

        For pyproject.toml and setup.cfg lists, entries have any extras
        specficiers (like ``[feature1,feature2]``) appended.
    """
    pyprojects: list[str] = []
    setups: list[str] = []
    requirements: list[str] = []

    for package in packages:
        name, extras = _split_extras(package)
        path = Path(name)

        if extras:
            joined_extras = f"[{','.join(extras)}]"
        else:
            joined_extras = ""

        pyproj_path = path / "pyproject.toml"
        if check_pyproject(pyproj_path):
            pyprojects.append(f"{pyproj_path}{joined_extras}")
            continue

        setup_path = path / "setup.cfg"
        if setup_path.exists():
            setups.append(f"{setup_path}{joined_extras}")
            continue

        requirements_path = path / "requirements.txt"
        if requirements_path.exists():
            requirements.append(str(requirements_path))
            continue

        raise ValueError(f"Could not process package: {package}")

    return pyprojects, setups, requirements


def get_python_deps(requirement: Requirement, python: str | None) -> tuple[list[Requirement], str]:
    """Get dependencies of a requirement"""
    tmpdir = tempfile.mkdtemp()
    try:
        req_str = str(requirement).partition(";")[0].strip()
        cmd = ["pip", "download", "--no-deps", "--only-binary=:all:", req_str]
        if python:
            cmd += ["--python-version", python]
        proc = run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"pip download {requirement} failed:\n{proc.stdout}\n{proc.stderr}"
            )

        wheels = list(Path(tmpdir).iterdir())
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one .whl file but found: {wheels}")
        # Note that extra requirements are processed just like the main
        # requirements but have an "extra" marker on them. Markers are difficult to
        # handle: https://github.com/pypa/packaging/issues/448
        wheel_file = pkginfo.Wheel(wheels[0])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # TODO: handle markers better. Here we just filter by extras and otherwise
    # drop markers
    new_reqs: list[Requirement] = []
    for req in wheel_file.requires_dist:
        if "extra" in req.partition(";")[-1]:
            for extra in requirement.extras:
                if 'extra == "{extra}"' in req:
                    new_reqs.append(Requirement(req.partition(";")[0]))
                    break
        else:
            new_reqs.append(Requirement(req))
    requires_python = wheel_file.requires_python or ""
    return new_reqs, requires_python


def classify_requirements_one_pass(
    requirements: list[Requirement],
    grayskull_map: YAML,
) -> tuple[list[tuple[Requirement, str]], list[Requirement]]:
    """Classify requirements as conda or pypi

    Args:
        requirements: list of PEP440 Python package specifications

    Returns:
        Two lists of tuples. The first contains conda package names and version
        specifiers and the second contains Python package names and specifiers.
    """
    proc = run(["conda", "info", "--json"], text=True, capture_output=True, check=True)
    channels = json.loads(proc.stdout)["channels"]

    subdirs = [conda.api.SubdirData(c) for c in channels]
    conda_reqs: list[tuple[Requirement, str]] = []
    python_reqs: list[Requirement] = []

    for req in requirements:
        matched = False
        match_name = req.name
        if match_name in grayskull_map and "conda_forge" in grayskull_map[match_name]:
            match_name = grayskull_map[match_name]["conda_forge"]
        # Handle pip's sloppiness about - and _
        replacements = [("", ""), ("_", "-"), ("-", "_")]
        for replacement, subdir in itertools.product(replacements, subdirs):
            match_name = match_name.replace(replacement[0], replacement[1])
            if subdir.query(match_name):
                matched = True
                break

        if matched:
            conda_reqs.append((req, match_name))
        else:
            python_reqs.append(req)

    return conda_reqs, python_reqs


def classify_requirements(
    requirements: list[str],
    python: str | None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """Classify requirements as conda or pypi

    Args:
        requirements: list of PEP440 Python package specifications
        python: Python version to use when inspecting subdependencies

    Returns:
        Two lists of tuples and a string. The first contains conda package
        names and version specifiers and the second contains Python package
        names and specifiers. The string contains the requires-python version
        specifier.
    """
    packaging_reqs = [Requirement(s) for s in requirements]
    yaml = YAML()
    grayskull_map = yaml.load(Path(PYPI_CONFIG).read_text())

    seen = set(packaging_reqs)

    conda_reqs, python_reqs = classify_requirements_one_pass(
        packaging_reqs, grayskull_map
    )

    sub_python_reqs = set(python_reqs)
    requires_python: list[str] = []

    while sub_python_reqs:
        req = sub_python_reqs.pop()
        req_deps, new_requires_python = get_python_deps(req, python)
        req_deps = [r for r in req_deps if r not in seen]
        new_conda_reqs, new_python_reqs = classify_requirements_one_pass(
            req_deps, grayskull_map
        )
        conda_reqs += new_conda_reqs
        python_reqs += new_python_reqs
        sub_python_reqs.update(new_python_reqs)
        if new_requires_python:
            requires_python.append(new_requires_python)
        seen.update(req_deps)

    conda_reqs.sort(key=lambda x: x[1].lower())
    python_reqs.sort(key=lambda x: x.name.lower())

    conda_table: dict[str, SpecifierSet] = {}
    for req, name in conda_reqs:
        name = name.lower()
        conda_table.setdefault(name, req.specifier)
        conda_table[name] = req.specifier & conda_table[name]

    canonical_re = re.compile(r"[-_.]+")
    python_table: dict[str, tuple[set[str], SpecifierSet]] = {}
    for req in python_reqs:
        name = canonical_re.sub("-", req.name.lower())
        python_table.setdefault(name, (req.extras, req.specifier))
        python_table[name] = (
            python_table[name][0] | req.extras,
            python_table[name][1] & req.specifier,
        )

    conda_tuples: list[tuple[str, str]] = []
    for name in sorted(conda_table):
        specifiers = conda_table[name]
        specs = str(specifiers).split(",")
        for idx, spec in enumerate(specs):
            if spec.startswith("~="):
                version = spec[len("~=") :]
                next_version = version.split(".")
                next_version[-1] = "0"
                next_version[-2] = str(int(next_version[-2]) + 1)
                specs[idx] = f">={version},<{'.'.join(next_version)}"

        conda_tuples.append((name, ",".join(specs)))

    python_tuples: list[tuple[str, str]] = []
    for name in sorted(python_table):
        extras, specifiers = python_table[name]
        extras_str = f"[{','.join(extras)}]" if req.extras else ""
        python_tuples.append((f"{name}{extras_str}", str(specifiers)))

    return conda_tuples, python_tuples, ",".join(requires_python)


def prepare_editable_requirements(
    editable_paths: list[str], pixi_path: Path
) -> list[str]:
    """Convert paths to editable projets into editable pip specs"""
    editables_name_path: list[tuple[str, Path]] = []
    for spec in editable_paths:
        if " @ " in spec:
            name, _, path_name = spec.partition(" @ ")
            name, _ = _split_extras(name.strip())
            path_name = path_name.strip()
            editables_name_path.append((name, Path(path_name)))
            continue

        path_name, _ = _split_extras(spec)

        path = Path(path_name)

        pyproj_path = path / "pyproject.toml"
        if pyproj_path.exists():
            config = tomllib.loads(pyproj_path.read_text())
            name = config.get("project", {}).get("name", "")
            if name:
                editables_name_path.append((name, path))
                continue

        setup_path = path / "setup.cfg"
        if setup_path.exists():
            config = configparser.ConfigParser()
            config.read(setup_path)
            if "metadata" in config and "name" in config["metadata"]:
                name = config["metadata"]["name"]
                editables_name_path.append((name, path))
                continue

        raise ValueError(f"Could not determine name for editable install: {path_name}")

    editable_specs: list[str] = []

    for name, path in editables_name_path:
        if path.is_absolute():
            editable_specs.append(f"{name} @ {path}")

        rel_path = path.relative_to(pixi_path.parent)
        editable_specs.append(f"{name} @ file://{EDITABLE_PLACEHOLDER_PATH}{rel_path}")

    return editable_specs


def update_pixi_toml(
    conda_requirements: list[tuple[str, str]],
    python_requirements: list[tuple[str, str]],
    editable_requirements: list[str],
    pixi_file: str,
    clear: bool = False,
):
    """Update pixi.toml file"""
    pixi_path = Path(pixi_file)

    if not pixi_path.exists():
        raise ValueError(f"Pixi file not found: {pixi_file}")

    if clear:
        config = tomllib.loads(pixi_path.read_text())
        old_conda = config.get("dependencies", {})
        old_pypi = config.get("pypi-dependencies", {})

        if old_conda or old_pypi:
            print("Clearing out old dependencies:\n")

        for pkgs in itertools.batched(old_pypi, 20):
            run(
                [
                    "pixi",
                    "remove",
                    "--manifest-path",
                    pixi_path,
                    "--no-lockfile-update",
                    "--pypi",
                ]
                + list(pkgs),
                check=True,
            )

        for pkgs in itertools.batched(old_conda, 20):
            run(
                ["pixi", "remove", "--manifest-path", pixi_path, "--no-lockfile-update"]
                + list(pkgs),
                check=True,
            )

    for pkgs in itertools.batched(conda_requirements, 20):
        run(
            ["pixi", "add", "--manifest-path", pixi_path, "--no-lockfile-update"]
            + [f"{name}{spec}" for name, spec in pkgs],
            check=True,
        )

    for pkgs in itertools.batched(python_requirements, 20):
        run(
            [
                "pixi",
                "add",
                "--pypi",
                "--manifest-path",
                pixi_path,
                "--no-lockfile-update",
            ]
            + [f"{name}{spec}" for name, spec in pkgs],
            check=True,
        )

    for pkgs in itertools.batched(editable_requirements, 20):
        run(
            [
                "pixi",
                "add",
                "--pypi",
                "--editable",
                "--manifest-path",
                pixi_path,
                "--no-lockfile-update",
            ]
            + list(pkgs),
            check=True,
        )

    if editable_requirements:
        pixi_config = pixi_path.read_text().splitlines()
        # Special case "." so it does not become "" because pixi would have removed a trailing "."
        pixi_config = [
            line.replace(f'"{EDITABLE_PLACEHOLDER_PATH}"', '"."') for line in pixi_config
        ]
        pixi_config = [
            line.replace(EDITABLE_PLACEHOLDER_PATH, "") for line in pixi_config
        ]
        pixi_path.write_text("\n".join(pixi_config))


def add_pixi_dependencies(
    editables: list[str] | None = None,
    packages: list[str] | None = None,
    requirements: list[str] | None = None,
    pyprojects: list[str] | None = None,
    setups: list[str] | None = None,
    condas: list[str] | None = None,
    python: str | None = None,
    pixi: str = "pixi.toml",
    clear: bool = False,
    print_only: bool = False,
):
    """Gather dependencies from inputs and insert into pixi.toml"""
    if not any((editables, packages, requirements, pyprojects, setups)):
        raise ValueError("No valid dependency inputs provided!")

    editables = editables or []
    packages = packages or []
    requirements = requirements or []
    pyprojects = pyprojects or []
    setups = setups or []
    condas = condas or []

    copy_editables_to_packages(editables, packages)

    pack_pyprojects, pack_setups, pack_requirements = detect_packages(packages)

    pyprojects = list(pyprojects) + pack_pyprojects
    setups = list(setups) + pack_setups
    requirements = list(requirements) + pack_requirements

    # Merged PEP508 requirement specifiers
    req_specifiers: list[str] = []
    requires_python: list[str] = []

    new_reqs, new_requires_python = gather_pyproject_reqs(pyprojects)
    req_specifiers.extend(new_reqs)
    if new_requires_python:
        requires_python.append(new_requires_python)

    new_reqs, new_requires_python = gather_setuptools_reqs(setups)
    req_specifiers.extend(new_reqs)
    if new_requires_python:
        requires_python.append(new_requires_python)

    req_specifiers += gatther_requirements_reqs(requirements)

    conda_reqs, python_reqs, new_requires_python = classify_requirements(req_specifiers, python)
    if new_requires_python:
        requires_python.append(new_requires_python)
    requires_python = {part for specs in requires_python for part in specs.split(",")}
    for req in condas:
        split_req = req.partition(" ")
        conda_reqs.append((split_req[0], split_req[2]))
    if python:
        requires_python.add(f"={python}")
    conda_reqs.append(("python", ",".join(requires_python)))
    editable_reqs = prepare_editable_requirements(editables, Path(pixi))

    if print_only:
        print("Conda requirements:")
        for spec in conda_reqs:
            print(" ".join(spec))
        print("\nPython requirements:")
        for spec in python_reqs:
            print(" ".join(spec))
    else:
        update_pixi_toml(conda_reqs, python_reqs, editable_reqs, pixi, clear)


def main(raw_args: list[str] | None = None):
    """Main CLI entry point"""
    parser = ArgumentParser(
        description="Generate pixi dependencies from Python dependencies"
    )
    parser.add_argument(
        "--editable",
        "-e",
        dest="editables",
        action="append",
        type=str,
        help=(
            "Like --package, but the path is added as a --pypi editable install "
            "as well as adding its dependencies"
        ),
    )
    parser.add_argument(
        "--package",
        "-p",
        dest="packages",
        action="append",
        type=str,
        help="Path to a Python package root directory to add the dependencies of",
    )
    parser.add_argument(
        "--requirements",
        "-r",
        action="append",
        type=str,
        help="Path to a requirements.txt file",
    )
    parser.add_argument(
        "--pyproject",
        "-t",
        dest="pyprojects",
        action="append",
        type=str,
        help="Path to a pyproject.toml file",
    )
    parser.add_argument(
        "--setup",
        "-s",
        dest="setups",
        action="append",
        type=str,
        help="Path to a setup.cfg file",
    )
    parser.add_argument(
        "--conda",
        dest="condas",
        action="append",
        type=str,
        help=(
            "Individual conda requirements to add. Use a space to add version "
            "bound like 'root 6.28'"
        ),
    )
    parser.add_argument(
        "--python",
        dest="python",
        type=str,
        help="Specific Python version to use",
    )
    parser.add_argument(
        "--pixi",
        "-P",
        default="pixi.toml",
        help="pixi.toml file to insert dependencies into",
    )
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear previous dependencies in pixi.toml before inserting",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Print out processed dependency list instead of modifying pixi.toml",
    )
    args = parser.parse_args(raw_args)

    add_pixi_dependencies(
        editables=args.editables,
        packages=args.packages,
        requirements=args.requirements,
        pyprojects=args.pyprojects,
        setups=args.setups,
        condas=args.condas,
        python=args.python,
        pixi=args.pixi,
        clear=args.clear,
        print_only=args.debug,
    )
