# SpinLinearChain

## Overview
This project consists on simualting a linear chain of spins with Heisenberg interaction.

## Goals
Simulation of observables that can be measured experimentally by tuning the different parameters of the system.

## Getting started

### Requirements

- Python 3.10 or newer
- Git

### Set up the environment (Windows PowerShell)

From the repository root, create and activate a virtual environment, then install the project and its development dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The virtual environment keeps this project's Python packages separate from other projects. The editable install makes the local package importable while you develop it; `[dev]` also installs tools used for development, currently `pytest`.

### Run the tests

```powershell
python -m pytest -q
```

The test suite checks the simulation components and expected physical behavior. Run it from the repository root after activating `.venv`.

## Project structure
To be writen

## Licence
TO be decided