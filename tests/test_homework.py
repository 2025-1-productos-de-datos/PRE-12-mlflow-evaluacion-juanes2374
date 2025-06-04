"""Grading"""

import os
import subprocess
import warnings
import sys

warnings.filterwarnings("ignore")


def test_01():
    # Ruta completa al ejecutable de Python dentro del entorno virtual
    python_exec = sys.executable

    # Test if the homework script runs without errors
    try:
        for model in ["elasticnet", "knn"]:
            subprocess.run(
                [python_exec, "-m", "homework", "--model", model],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running the homework script: {e}")

    # Ensure the mlruns directory exists
    assert os.path.exists("mlruns"), "mlruns directory does not exist."

    # Check if there are any experiments saved in mlruns/
    experiments = [
        d for d in os.listdir("mlruns") if os.path.isdir(os.path.join("mlruns", d))
    ]
    assert len(experiments) > 0, "No experiments found in mlruns directory."
