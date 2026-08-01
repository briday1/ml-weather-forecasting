from __future__ import annotations

import ast
from pathlib import Path


def test_experiment_is_a_linear_notebook_style_script():
    source = Path("experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "argparse" not in imports
    assert "parse_args" not in function_names
    assert "main" not in function_names
    assert "# %% Experiment settings" in source
    assert "# %% Train" in source
    assert "# %% Evaluate" in source
    assert "# %% Plot" in source
