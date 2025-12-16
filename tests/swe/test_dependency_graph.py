import pytest
from hologram.container import HologramContainer
from hologram.swe.dependency_graph import CodeDependencyGraph


@pytest.fixture
def fact_store_with_deps():
    container = HologramContainer(dimensions=1000)
    fs = container.create_fact_store()
    # Add call graph facts
    fs.add_fact("process", "module", "utils.py")
    fs.add_fact("validate", "module", "helpers.py")
    fs.add_fact("process", "calls", "validate")
    fs.add_fact("validate", "called_by", "process")
    return fs


def test_bfs_finds_callers(fact_store_with_deps):
    graph = CodeDependencyGraph(fact_store_with_deps)
    result = graph.get_affected_files("helpers.py")
    assert "utils.py" in result.affected_files


def test_respects_max_depth(fact_store_with_deps):
    graph = CodeDependencyGraph(fact_store_with_deps, max_depth=1)
    result = graph.get_affected_files("helpers.py")
    assert result.total_depth == 1
