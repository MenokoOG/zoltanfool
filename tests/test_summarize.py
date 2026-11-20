import json

from zoltanfool.summarize import pick, summarize


def test_worked_example_mean_and_range(tmp_path):
    for i, v in enumerate((1.92, 1.95, 1.90)):
        (tmp_path / f"a{i}.json").write_text(json.dumps({"label": "A", "bits_per_byte": v}))
    (tmp_path / "rft.json").write_text(json.dumps({"label": "rft", "heldout_after": {}}))
    [(label, n, mean, lo, hi)] = summarize(str(tmp_path))
    assert (label, n) == ("A", 3)
    assert abs(mean - 1.923333) < 1e-5 and abs((hi - lo) - 0.05) < 1e-9


def test_pick_dotted_key_with_decimal():
    assert pick({"syntax": {"0.8": 0.4}}, "syntax.0.8") == 0.4
    assert pick({"syntax": {}}, "syntax.0.8") is None
