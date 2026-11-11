from dataclasses import dataclass

import numpy as np
import pytest

from eval.metrics import action_map, coverage, highlight_f1, label_accuracy, score_all


def action(start=0, end=10, verb="cut", noun="tomato", score=1):
    return dict(start=start, end=end, verb=verb, noun=noun, score=score)


def highlight(start, end):
    return dict(start=start, end=end)


def test_coverage_counts_distinct_pairs_and_any_instance():
    gt = [action(), action(20, 30), action(40, 50, "wash")]
    assert coverage(gt, [highlight(20, 30)], "any") == 0.5
    assert coverage(gt, [], "any") == 0
    assert coverage([], [], "any") == 0


def test_coverage_union_and_half_open_boundaries():
    gt = [action()]
    assert coverage(gt, [highlight(0, 3), highlight(2, 4)], "frac>=0.5") == 0
    assert coverage(gt, [highlight(0, 2), highlight(6, 9)], "frac>=0.5") == 1
    assert coverage(gt, [highlight(10, 11)], "any") == 0
    assert coverage(gt, [highlight(0, 5)], "centre") == 0
    assert coverage(gt, [highlight(5, 6)], "centre") == 1


@pytest.mark.parametrize("rule", ["unknown", "frac>=0", "frac>=nan", "frac>=1.1"])
def test_coverage_requires_valid_explicit_rule(rule):
    with pytest.raises(ValueError):
        coverage([], [], rule)


def test_map_perfect_empty_and_missing_class():
    gt = [action(), action(20, 30, "wash", "dish")]
    for kind in ("pair", "verb", "noun"):
        assert action_map(gt, gt)[kind]["mAP"] == 1
        assert action_map(gt, [])[kind]["mAP"] == 0
        assert action_map([], [])[kind]["mAP"] == 0
        assert action_map(gt, gt[:1])[kind]["mAP"] == 0.5


def test_map_ranking_and_duplicate_lock():
    gt = [action(0, 10), action(20, 30)]
    pred = [action(0, 10, score=1), action(0, 10, score=0.9), action(20, 30, score=0.8)]
    assert action_map(gt, pred)["pair"]["mAP"] == pytest.approx(5 / 6)
    assert action_map(gt[:1], [action(40, 50, score=1), action(score=0.5)])["pair"]["mAP"] == 0.5


def test_map_video_ids_prevent_cross_video_match_and_pool_scores():
    gt = {"a": [action()], "b": [action()]}
    pred = {"wrong": [action(score=1)], "a": [action(score=0.5)], "b": []}
    assert action_map(gt, pred)["pair"]["mAP"] == 0.25


def test_map_thresholds_and_label_components():
    result = action_map([action()], [action(0, 2)], tious=[0.1, 0.3])
    assert result["pair"]["per_tiou"] == [1, 0]
    result = action_map([action()], [action(noun="onion")])
    assert result["pair"]["mAP"] == result["noun"]["mAP"] == 0
    assert result["verb"]["mAP"] == 1


def test_map_ties_are_stable():
    pred = [action(30, 40, score=1), action(score=1)]
    assert action_map([action()], pred)["pair"]["mAP"] == 0.5
    assert action_map([action()], pred[::-1])["pair"]["mAP"] == 1


@pytest.mark.parametrize("tious", [[], [0], [-1], [float("nan")], [1.1]])
def test_invalid_tious(tious):
    with pytest.raises(ValueError):
        action_map([], [], tious)


def test_confidence_not_silently_invented():
    pred = action()
    del pred["score"]
    with pytest.raises(KeyError):
        action_map([action()], [pred])
    with pytest.raises(ValueError):
        action_map([action()], [action(score=float("nan"))])
    with pytest.raises(ValueError):
        action_map({"a": []}, [])


def test_highlight_f1_known_answers():
    times = np.arange(20) / 10
    imp = np.zeros(20)
    imp[:3] = 1
    assert highlight_f1(imp, times, [highlight(0, 0.3)]) == dict(precision=1, recall=1, f1=1)
    assert highlight_f1(imp, times, [highlight(0, 0.6)])["f1"] == pytest.approx(2 / 3)
    assert highlight_f1(imp, times, [])["f1"] == 0
    assert highlight_f1([], [], [])["f1"] == 0


def test_f1_rank_rounding_ties_and_union():
    assert highlight_f1([0] * 10, np.arange(10), [highlight(0, 2)])["f1"] == 1
    assert highlight_f1([1, 0], [0, 1], [highlight(0, 1), highlight(0, 1)])["f1"] == 1


@pytest.mark.parametrize("values,times", [([1], [0, 1]), ([1, 0], [0, 0]),
                                        ([1], [-1]), ([float("nan")], [0]), ([2], [0])])
def test_f1_bad_grids(values, times):
    with pytest.raises(ValueError):
        highlight_f1(values, times, [])


def test_label_accuracy_alignment():
    assert label_accuracy(["a", "b"], ["a", "b"]) == 1
    assert label_accuracy({"a": "x", "b": "y"}, {"a": "x"}) == 0.5
    assert label_accuracy([], []) == 0
    with pytest.raises(ValueError):
        label_accuracy(["a"], [])
    with pytest.raises(ValueError):
        label_accuracy({"a": "x"}, {"unknown": "x"})


def test_dataclasses_supported():
    @dataclass
    class Action:
        start: float = 0
        end: float = 1
        verb: str = "cut"
        noun: str = "tomato"
        score: float = 1
    assert action_map([Action()], [Action()])["pair"]["mAP"] == 1
    assert coverage([Action()], [Action()], "any") == 1


def test_score_all_no_fabricated_missing_metrics_or_validity():
    args = dict(duration=100, gt_actions=[action(0, 15)], pred_actions=[action(0, 15)],
                highlights=[highlight(0, 15)], coverage_rule="frac>=0.5")
    report = score_all(**args)
    assert report["weighted_total_proxy"] is None
    args.update(times=np.arange(100), human_importance=[1] * 15 + [0] * 85,
                gt_labels=["cooking"], pred_labels=["cooking"])
    assert score_all(**args)["weighted_total_proxy"] == 1
    args["highlights"] = [highlight(0, 16)]
    report = score_all(**args)
    assert not report["budget_ok"] and report["weighted_total_proxy"] is None


def test_official_segment_boundary_is_supported_without_guessing():
    gt = [dict(segment=[0, 10], verb="cut", noun="tomato", score=.9)]
    hls = [dict(segment=[0, 10])]
    assert coverage(gt, hls, "frac>=0.5") == 1
    assert action_map(gt, gt)["pair"]["mAP"] == 1
    from eval.metrics import budget_ok
    assert budget_ok(hls, 100)
    with pytest.raises(ValueError, match="Ambiguous"):
        coverage(gt, [dict(segment=[0, 10], start=5)], "any")


def test_ablation_uses_dataset_metrics_and_labels_synthetic_evidence():
    from scripts.run_ablation import synthetic_cases, run_cases
    result = run_cases(synthetic_cases(), coverage_rule="frac>=0.5", synthetic=True)
    assert result["synthetic_only"] and not result["official_metrics_verified"]
    assert result["selector_inputs"] == "predicted"
    assert set(result["summary"]) == {"U", "I", "A", "J"}
    assert all(r["strict_budget_violations"] == 0 for r in result["summary"].values())
    assert result["summary"]["J"]["coverage_proxy"] == 1
    assert result["summary"]["U"]["coverage_proxy"] < 1
    assert result["summary"]["J"]["f1_proxy"] > result["summary"]["U"]["f1_proxy"]
    assert all(r["f1_proxy"] is None for r in result["records"] if r["dataset"] == "epic_kitchens_100")


def test_ablation_missing_metrics_cannot_receive_combined_total():
    from scripts.run_ablation import synthetic_cases, run_cases
    cases = list(synthetic_cases())
    cases[1]["human_times"] = cases[1]["human_importance"] = None
    result = run_cases(cases, coverage_rule="frac>=0.5", synthetic=True, oracle=True)
    assert result["selector_inputs"] == "oracle_gt"
    assert all(r["weighted_total_proxy"] is None for r in result["summary"].values())


def test_real_study_loader_rejects_evaluation_and_unreviewed_inputs(tmp_path):
    import json
    from perception_cache import write_cache, file_sha256
    from scripts.run_ablation import synthetic_cases, load_study
    case = next(synthetic_cases())
    case["info"].video_id = "TEST-ONLY-fixture"
    source = tmp_path/"synthetic-source.bin"
    source.write_bytes(b"TEST ONLY: no genuine video or checkpoint")
    case["info"].path = str(source)
    manifest = write_cache(tmp_path/"cache", case["info"], case["clip"], case["imp"], {"purpose": "unit-test"})
    predictions, annotations = tmp_path/"pred.json", tmp_path/"gt.json"
    predictions.write_text(json.dumps({"source": "predicted", "input_sha256": file_sha256(source),
                                       "checkpoint_sha256": "a"*64, "actions": []}))
    annotations.write_text(json.dumps({"input_sha256": file_sha256(source), "actions": []}))
    study = tmp_path/"study.json"
    study.write_text(json.dumps({"format": "activa-m3-study-v1", "videos": [
        {"dataset": "epic_kitchens_100", "m2_manifest": str(manifest),
         "predictions": str(predictions), "annotations": str(annotations)}]}))
    official, review = tmp_path/"manifest.csv", tmp_path/"review.json"
    official.write_text("video_id,duration_sec,dataset\nevaluation-fixture,120,epic_kitchens_100\n")
    # Explicitly fictional unit-test approval; never used for real-data clearance.
    decision = {"format": "activa-leakage-review-v1", "state": "reviewed",
                "reviewed_by": "TEST-ONLY", "reviewed_at": "TEST-ONLY",
                "official_manifest_sha256": file_sha256(official), "videos": [
                    {"video_id": case["info"].video_id, "status": "reviewed-safe",
                     "sha256": file_sha256(source), "evidence": "Fictional unit fixture"}]}
    review.write_text(json.dumps(decision))
    assert len(list(load_study(study, review, official))) == 1
    decision["videos"][0]["status"] = "unresolved"
    review.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="unreviewed"):
        list(load_study(study, review, official))
    decision["videos"][0]["status"] = "reviewed-safe"
    official.write_text("video_id,duration_sec,dataset\nTEST-ONLY-fixture,120,epic_kitchens_100\n")
    decision["official_manifest_sha256"] = file_sha256(official)
    review.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="evaluation-overlap"):
        list(load_study(study, review, official))
