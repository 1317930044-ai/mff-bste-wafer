import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

from main import parser_for_cli
from wafer.common import BRANCHES, CONFIG, FEATURE_VERSION, ROOT, RUNS, Progress, load_config, read_csv, read_json
from wafer.data import decode, export_png, parse_label, read_map, synthetic_map, validate_map
from wafer.experiment import design, evaluate, metrics
from wafer.features import Wafer, extract, projections, sinogram
from wafer.inference import load_model, predict, predict_directory
from wafer.models import fit_stack, oof_folds
from wafer.preparation import split_indices


class TestCore(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.settings = self.config["modes"]["quick"]

    def test_nested_and_missing_labels(self):
        cases = [(np.array([["Scratch"]]), "Scratch"), ([[b"edge_ring"]], "Edge-Ring"), (" none ", "None"),
                 (np.array([]), None), (None, None), (float("nan"), None), ("mystery", None)]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_label(value), expected)

    def test_map_formats_and_rejection(self):
        a = np.array([[0, 1, 2], [2, 2, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            png, npy, npz = (base / f"sample.{suffix}" for suffix in ("png", "npy", "npz"))
            export_png(a, png)
            np.save(npy, a)
            np.savez_compressed(npz, wafer=a)
            for file in (png, npy, npz):
                np.testing.assert_array_equal(read_map(file), a)
            wrong = base / "wrong.png"
            Image.new("RGB", (3, 2), (1, 1, 1)).save(wrong)
            with self.assertRaises(ValueError):
                read_map(wrong)
        buf = io.BytesIO()
        np.save(buf, np.array([{"pickle": True}], dtype=object))
        with self.assertRaises(ValueError):
            decode(buf.getvalue(), "unsafe.npy")
        with self.assertRaises(ValueError):
            decode(b"fake", "photo.jpg")
        for invalid in (np.array([[0]]), np.array([[3]]), np.array([[1.2]]), np.array([1, 2]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                validate_map(invalid)

    def test_fixed_lengths_and_finite_values(self):
        data = (np.array([[1]]), np.array([[2]]), np.ones((5, 12)), np.full((16, 3), 2),
                np.array([[0, 1, 0], [2, 1, 2]]), synthetic_map("Scratch", np.random.default_rng(4)))
        expected = {"spatial": 28, "geometry": 15, "radon": 20, "texture": 14, "projection": 16}
        for a in data:
            _, groups, times = extract(a, self.settings)
            self.assertEqual({name: len(values) for name, values in groups.items()}, expected)
            self.assertTrue(all(np.isfinite(list(values.values())).all() for values in groups.values()))
            self.assertEqual(set(times), {"mask", *BRANCHES})
        self.assertEqual(len(extract(data[0], self.config["modes"]["full"])[1]["radon"]), 26)
        self.assertNotIn("radon", extract(data[0], self.settings, ("spatial",))[1])

    def test_valid_denominator_and_radon_conservation(self):
        a = np.array([[0, 0, 0], [0, 2, 1], [0, 0, 0]], dtype=np.uint8)
        w, features, _ = extract(a, self.settings)
        self.assertEqual(features["spatial"]["spatial.defect_share"], .5)
        rows, _ = tuple(projections(w))
        self.assertEqual(rows[2].tolist(), [0, .5, 0])
        rays, valid = sinogram(w, self.settings)
        self.assertTrue(np.all(rays <= valid))
        self.assertTrue(np.all(rays.sum(axis=1) == rays[0].sum()))

    def test_split_reproducible_and_disjoint(self):
        lots = np.repeat([f"lot{i:02}" for i in range(30)], 4)
        y = np.tile([0, 0, 1, 2], 30)
        for strategy in ("stratified", "grouped"):
            first = split_indices(y, lots, [.6, .2, .2], 42, strategy)
            again = split_indices(y, lots, [.6, .2, .2], 42, strategy)
            sets = [set(ix) for ix in first.values()]
            self.assertEqual(len(set.union(*sets)), len(y))
            self.assertEqual(sum(map(len, sets)), len(y))
            for key in first:
                np.testing.assert_array_equal(first[key], again[key])
                self.assertEqual(set(y[first[key]]), set(y))
            if strategy == "grouped":
                group_sets = [set(lots[ix]) for ix in first.values()]
                self.assertEqual(sum(map(len, group_sets)), len(set(lots)))

    def test_oof_groups_and_probabilities(self):
        rng = np.random.default_rng(12)
        y = np.tile([0, 1, 2], 18)
        lots = np.repeat([f"l{i}" for i in range(18)], 3)
        ids = np.array([f"id{i}" for i in range(len(y))])
        folds = oof_folds(y, lots, True, 3, 42)
        for fit, hold in folds:
            self.assertFalse(set(fit) & set(hold))
            self.assertFalse(set(lots[fit]) & set(lots[hold]))
        x = rng.normal(size=(len(y), 6)) + y[:, None]
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            model, manifest = fit_stack(x, y, ids, lots, True, self.config,
                                        dict(self.settings, n_estimators=3, boost_iterations=4), Progress(quiet=True))
        self.assertEqual(len({r["id"] for r in manifest}), len(y))
        self.assertTrue(all(np.isclose(sum(r[f"random_forest_p{i}"] for i in range(3)), 1) for r in manifest))
        self.assertTrue(np.allclose(model.predict_proba(x[:2]).sum(axis=1), 1))

    def test_model_roundtrip_and_batch(self):
        a = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        branches = ("spatial",)
        _, groups, _ = extract(a, self.settings, branches)
        features = groups["spatial"]
        row = list(features.values())
        model = DecisionTreeClassifier(random_state=42).fit([row, [0]*len(row)], [0, 1])
        bundle = dict(model=model, name="fixture", classes=["Center", "Scratch"], task="defect8", source="synthetic",
                      run_id="test", feature_version=FEATURE_VERSION, branches=branches, settings=self.settings, names=list(features))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            png, npy, destination = base / "a.png", base / "b.npy", base / "model.joblib"
            export_png(a, png)
            np.save(npy, a)
            joblib.dump(bundle, destination)
            _, before = predict(bundle, path=png)
            _, after = predict(load_model(destination), path=npy)
            self.assertEqual(before["prediction"], after["prediction"])
            self.assertEqual(before["probability"], after["probability"])
            self.assertEqual(before["features"], after["features"])
            (base / "invalid.jpg").write_bytes(b"invalid")
            rows = predict_directory(bundle, base, base / "batch.csv", Progress(quiet=True))
            self.assertEqual(sum(not r["error"] for r in rows), 2)

    def test_cli_and_experiment_matrix(self):
        parser = parser_for_cli()
        with self.assertRaises(SystemExit) as exit_:
            parser.parse_args(["--help"])
        self.assertEqual(exit_.exception.code, 0)
        specs = design(SimpleNamespace(command="train-all", mode="standard", external=False), BRANCHES)
        self.assertEqual(sum(spec.group == "features" for spec in specs), 10)
        self.assertGreaterEqual(sum(spec.group == "model" for spec in specs), 7)

    def test_metrics_and_frozen_run(self):
        stats, rows, matrix = metrics([0, 0, 1], [0, 0, 0], ["A", "B"])
        self.assertAlmostEqual(stats["macro_f1"], .4)
        self.assertEqual(matrix, [[2, 0], [1, 0]])
        self.assertEqual(rows[1]["recall"], 0)
        run = RUNS / "demo-synthetic"
        if (run / "frozen.json").exists():
            self.assertEqual(evaluate(run)[0]["model"], read_json(run / "frozen.json")["best"])
            split = read_csv(run / "split_manifest.csv")
            train = {row["id"] for row in split if row["part"] == "train"}
            self.assertEqual(train, {row["id"] for row in read_csv(run / "oof.csv")})


if __name__ == "__main__":
    unittest.main()
