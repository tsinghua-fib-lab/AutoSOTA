from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister.training.evaluate_predictions import evaluate_predictions


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class EvaluatePredictionsTests(unittest.TestCase):
    def _make_predictions_payload(self) -> dict:
        return {
            "config": {
                "dataset_id": "001",
                "dataset_name": "Dataset_001_Test",
            },
            "cases": {
                "case_001": {
                    "rois": {
                        "left": {
                            "probabilities": [0.90, 0.05, 0.05],
                            "prediction": 2,
                        },
                        "right": {
                            "probabilities": [0.10, 0.80, 0.10],
                            "prediction": 0,
                        },
                    }
                },
                "case_002": {
                    "rois": {
                        "left": {
                            "probabilities": [0.10, 0.20, 0.70],
                            "prediction": 1,
                        },
                        "right": {
                            "probabilities": [0.60, 0.20, 0.20],
                            "prediction": 1,
                        },
                    }
                },
                "case_003": {
                    "rois": {
                        "left": {
                            "probabilities": [0.20, 0.60, 0.20],
                            "prediction": 0,
                        },
                        "right": {
                            "probabilities": [0.15, 0.20, 0.65],
                            "prediction": 0,
                        },
                    }
                },
            },
        }

    def test_evaluate_predictions_writes_json_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(
                targets_path,
                {
                    "case_001_left": 0,
                    "case_001_right": 1,
                    "case_002_left": 2,
                    "case_002_right": 0,
                    "case_003_left": 1,
                    "case_003_right": 2,
                },
            )
            _write_json(predictions_path, self._make_predictions_payload())

            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                output_path = evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=32,
                    seed=7,
                )

            self.assertEqual(output_path, root / "evaluation.json")
            self.assertTrue((root / "evaluation.json").is_file())
            self.assertTrue((root / "confusion_matrix.png").is_file())
            self.assertTrue((root / "macro_auc_curve.png").is_file())

            payload = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["num_samples"], 6)
            self.assertIn("extended_summary", payload)
            self.assertIn("challenge_metrics", payload)
            self.assertEqual(
                payload["extended_summary"]["confusion_matrix"],
                [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            )
            self.assertEqual(payload["predictions"]["case_001_left"]["prediction"], 0)
            self.assertIsNotNone(payload["challenge_metrics"]["macro_auc"])
            self.assertIsNotNone(
                payload["challenge_metrics"]["macro_specificity_at_90_sensitivity"]
            )
            self.assertIsNotNone(
                payload["challenge_metrics"]["macro_sensitivity_at_90_specificity"]
            )
            self.assertIn("Confusion matrix", mock_stdout.getvalue())
            self.assertIn("macro_auc", mock_stdout.getvalue())
            self.assertIn("specificity_at_90_sensitivity", mock_stdout.getvalue())
            self.assertIn("sensitivity_at_90_specificity", mock_stdout.getvalue())
            self.assertNotIn("rank_score", mock_stdout.getvalue())

    def test_evaluate_predictions_accepts_list_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(
                targets_path,
                {
                    "case_001_left": [1, 0, 0],
                    "case_001_right": [0, 1, 0],
                    "case_002_left": [0, 0, 1],
                },
            )
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.7, 0.2, 0.1]},
                                "right": {"probabilities": [0.1, 0.8, 0.1]},
                            }
                        },
                        "case_002": {
                            "rois": {
                                "left": {"probabilities": [0.1, 0.2, 0.7]},
                            }
                        },
                    }
                },
            )

            output_path = evaluate_predictions(
                targets_path=str(targets_path),
                predictions_path=str(predictions_path),
                output_path=str(root / "reports"),
                n_bootstrap=16,
            )

            self.assertEqual(output_path, root / "reports" / "evaluation.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["balanced_accuracy"], 1.0)

    def test_evaluate_predictions_uses_probability_argmax_instead_of_stored_prediction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(
                targets_path,
                {
                    "case_001_left": 2,
                    "case_002_left": 1,
                    "case_003_left": 0,
                },
            )
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {
                                    "probabilities": [0.1, 0.2, 0.7],
                                    "prediction": 0,
                                }
                            }
                        },
                        "case_002": {
                            "rois": {
                                "left": {
                                    "probabilities": [0.2, 0.6, 0.2],
                                    "prediction": 2,
                                }
                            }
                        },
                        "case_003": {
                            "rois": {
                                "left": {
                                    "probabilities": [0.8, 0.1, 0.1],
                                    "prediction": 1,
                                }
                            }
                        },
                    }
                },
            )

            output_path = evaluate_predictions(
                targets_path=str(targets_path),
                predictions_path=str(predictions_path),
                n_bootstrap=16,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["balanced_accuracy"], 1.0)
            self.assertEqual(payload["predictions"]["case_001_left"]["prediction"], 2)

    def test_evaluate_predictions_fails_for_missing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(
                targets_path,
                {
                    "case_001_left": 0,
                    "case_999_right": 1,
                },
            )
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                            }
                        }
                    }
                },
            )

            output_path = evaluate_predictions(
                targets_path=str(targets_path),
                predictions_path=str(predictions_path),
                n_bootstrap=8,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["num_samples"], 1)

    def test_evaluate_predictions_fails_for_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(
                targets_path,
                {"case_001_left": 0},
            )
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                                "right": {"probabilities": [0.1, 0.8, 0.1]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(
                ValueError, "Missing targets for predicted samples"
            ):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_fails_for_non_three_class_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {"case_001_left": 0})
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.2]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "must contain exactly 3 values"):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_fails_for_out_of_range_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {"case_001_left": 3})
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "must be one of 0, 1, 2"):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_rejects_boolean_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {"case_001_left": True})
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(TypeError, "Boolean labels are not supported"):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_rejects_empty_targets_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {})
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "Target JSON must not be empty"):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_rejects_non_finite_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {"case_001_left": 0})
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, "nan", 0.2]},
                            }
                        }
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "must be finite numbers"):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_rejects_missing_cases_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            _write_json(targets_path, {"case_001_left": 0})
            _write_json(predictions_path, {"config": {}})

            with self.assertRaisesRegex(
                ValueError,
                "predictions.json must define a 'cases' object",
            ):
                evaluate_predictions(
                    targets_path=str(targets_path),
                    predictions_path=str(predictions_path),
                    n_bootstrap=8,
                )

    def test_evaluate_predictions_accepts_json_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            targets_path = root / "labels.json"
            predictions_path = root / "predictions.json"
            output_path = root / "reports" / "custom_eval.json"
            _write_json(
                targets_path,
                {
                    "case_001_left": 0,
                    "case_001_right": 1,
                    "case_002_left": 2,
                },
            )
            _write_json(
                predictions_path,
                {
                    "cases": {
                        "case_001": {
                            "rois": {
                                "left": {"probabilities": [0.8, 0.1, 0.1]},
                                "right": {"probabilities": [0.1, 0.8, 0.1]},
                            }
                        },
                        "case_002": {
                            "rois": {
                                "left": {"probabilities": [0.1, 0.2, 0.7]},
                            }
                        },
                    }
                },
            )

            returned_path = evaluate_predictions(
                targets_path=str(targets_path),
                predictions_path=str(predictions_path),
                output_path=str(output_path),
                n_bootstrap=16,
            )

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertTrue((root / "reports" / "confusion_matrix.png").is_file())
            self.assertTrue((root / "reports" / "macro_auc_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
