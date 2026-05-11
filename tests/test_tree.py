from otllm.tree.node import ThoughtNode
from otllm.tree.thought_tree import ThoughtTree


def test_add_node_and_root():
    tree = ThoughtTree()
    root = ThoughtNode(id="root", text="first thought", context_summary="initial")
    tree.add_node(root)
    assert tree.root is root
    assert tree.node_count == 1


def test_parent_child():
    tree = ThoughtTree()
    root = ThoughtNode(id="root", text="root", context_summary="ctx")
    child = ThoughtNode(id="c1", parent_id="root", depth=1, text="child", context_summary="ctx2")
    tree.add_node(root)
    tree.add_node(child)
    assert tree.get_children("root") == [child]
    assert tree.get_ancestors("c1") == [root]


def test_path_to_root():
    tree = ThoughtTree()
    n0 = ThoughtNode(id="n0", text="a", context_summary="a")
    n1 = ThoughtNode(id="n1", parent_id="n0", depth=1, text="b", context_summary="b")
    n2 = ThoughtNode(id="n2", parent_id="n1", depth=2, text="c", context_summary="c")
    tree.add_node(n0)
    tree.add_node(n1)
    tree.add_node(n2)
    path = tree.get_path_to_root("n2")
    assert [n.id for n in path] == ["n0", "n1", "n2"]


def test_bfs_order():
    tree = ThoughtTree()
    root = ThoughtNode(id="r", text="r", context_summary="r")
    c1 = ThoughtNode(id="c1", parent_id="r", depth=1, text="c1", context_summary="c1")
    c2 = ThoughtNode(id="c2", parent_id="r", depth=1, text="c2", context_summary="c2")
    tree.add_node(root)
    tree.add_node(c1)
    tree.add_node(c2)
    ids = [n.id for n in tree.bfs()]
    assert ids == ["r", "c1", "c2"]


def test_leaves():
    tree = ThoughtTree()
    root = ThoughtNode(id="r", text="r", context_summary="r")
    c1 = ThoughtNode(id="c1", parent_id="r", depth=1, text="c1", context_summary="c1")
    tree.add_node(root)
    tree.add_node(c1)
    leaves = tree.get_leaves()
    assert len(leaves) == 1
    assert leaves[0].id == "c1"


def test_serialization_roundtrip():
    tree = ThoughtTree()
    root = ThoughtNode(id="r", text="root thought", context_summary="initial context")
    child = ThoughtNode(id="c", parent_id="r", depth=1, text="child thought", context_summary="updated")
    tree.add_node(root)
    tree.add_node(child)

    data = tree.to_dict()
    restored = ThoughtTree.from_dict(data)
    assert restored.node_count == 2
    assert restored.root.id == "r"
    assert restored.get_children("r")[0].id == "c"


def test_compressed_view():
    tree = ThoughtTree()
    root = ThoughtNode(id="r", text="root", context_summary="start", drift_from_anchor=0.0)
    child = ThoughtNode(id="c", parent_id="r", depth=1, text="child", context_summary="middle", drift_from_anchor=0.15)
    tree.add_node(root)
    tree.add_node(child)
    view = tree.compressed_view("c")
    assert "[0]" in view
    assert "[1]" in view
    assert "0.000" in view
    assert "0.150" in view
