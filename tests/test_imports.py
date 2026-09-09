"""Structural checks that catch a whole class of editing accident.

A careless refactor once deleted four functions from `report.py` by replacing an
over-wide slice, and nothing noticed until the report stage failed at the very end of a
long run. These tests are cheap and catch that immediately.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mindeye_lora"


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_every_intra_package_import_resolves():
    modules = {p.stem: _top_level_names(p) for p in SRC.glob("*.py")}
    problems = []
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.ImportFrom) and node.level == 1):
                continue
            if node.module not in modules:
                continue
            for alias in node.names:
                if alias.name not in modules[node.module]:
                    problems.append(f"{path.name} imports {alias.name!r} from {node.module}")
    assert not problems, "unresolved intra-package imports: " + "; ".join(problems)


def test_report_exposes_every_figure_function():
    """cli.cmd_report imports these by name; losing one breaks only the final stage."""
    names = _top_level_names(SRC / "report.py")
    for required in ("figure_metric_bars", "figure_forest", "figure_pareto",
                     "figure_training_curves", "figure_qualitative",
                     "figure_retrieval_grid", "select_qualitative_arms", "build_report"):
        assert required in names, f"report.py is missing {required}"


def test_no_module_lost_its_public_api():
    """Rough guard against a module being truncated: each should define something."""
    for path in sorted(SRC.glob("*.py")):
        if path.name == "__init__.py":
            continue
        assert _top_level_names(path), f"{path.name} defines nothing at top level"
