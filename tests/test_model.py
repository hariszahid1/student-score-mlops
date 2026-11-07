import json

PASS_MARK = 8.0  # maximum allowed MAE in exam points


def test_mae_is_below_pass_mark():
    with open("metrics.json") as f:
        mae = json.load(f)["mae"]
    assert mae < PASS_MARK, f"MAE {mae:.2f} is above the pass mark {PASS_MARK}"
