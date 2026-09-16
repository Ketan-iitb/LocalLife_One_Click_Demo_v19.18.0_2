"""Packaging descriptor so `pip install -e .` actually works.

Root cause of the v19.8.3 "does not appear to be a Python project: neither
'setup.py' nor 'pyproject.toml' found" crash: this project never shipped an
installable package descriptor at all -- only requirements-*.txt files. The
launcher's `pip install -e .` step was always going to fail the moment it
was genuinely reached. It went unnoticed through round 8.6 because the
launcher was crashing one step earlier, at the `pip show` stderr-promotion
bug fixed in v19.7.3 -- fixing that bug is what finally let a real run
reach this line and surface the gap. This file is the fix.
"""
from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).parent


def _read_requirements() -> list[str]:
    text = (_HERE / "requirements-local.txt").read_text(encoding="utf-8")
    reqs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="locallife-cloud",
    version="19.20.0",
    description="LocalLife dual-camera waste volume measurement (local, cloud-free runtime).",
    packages=find_packages(include=["locallife_cloud", "locallife_cloud.*"]),
    # recipe_config.yaml (the v3 recipe pipeline's own default tuning file,
    # locallife_cloud/recipe_config.py) and box_templates.yaml (round 16's
    # table-relative cuboid box templates, locallife_cloud/box_templates.py)
    # ship alongside the .py files -- this only matters for a real wheel
    # build; the launcher's `pip install -e .` editable install never copies
    # the source tree at all, so both modules' relative-path lookups already
    # find their YAML either way.
    package_data={"locallife_cloud": ["recipe_config.yaml", "box_templates.yaml"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=_read_requirements(),
)
