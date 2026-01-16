"""Tests for deploy domain atoms."""

from garbage_deploy.domain.atoms import (
    DetectionState,
    check_detection_stability,
    can_count_new_garbage,
    map_serial_payload,
)


def test_check_detection_stability_new_type():
    """New detection type should not be stable immediately."""
    state = DetectionState()
    new_state, stable = check_detection_stability(
        state, detected_type=0, now=100.0, min_position_frames=5
    )
    # New detection type resets the state and is not stable yet
    assert stable is False
    # State is reset when detection type changes
    assert new_state.detected_type is None
    assert new_state.position_frames == 0


def test_check_detection_stability_after_min_frames():
    """After reaching min_position_frames, detection should be stable."""
    state = DetectionState(detected_type=0, position_frames=4)
    new_state, stable = check_detection_stability(
        state, detected_type=0, now=100.0, min_position_frames=5
    )
    assert stable is True
    assert new_state.detected_type == 0
    assert new_state.position_frames == 5


def test_check_detection_stability_existing_type():
    """Continuing detection should increment frames."""
    state = DetectionState(detected_type=1, position_frames=5)
    new_state, stable = check_detection_stability(
        state, detected_type=1, now=100.0, min_position_frames=5
    )
    assert stable is True
    assert new_state.detected_type == 1
    assert new_state.position_frames == 6


def test_check_detection_stability_not_enough_frames():
    """Not enough frames should not be stable."""
    state = DetectionState(detected_type=1, position_frames=3)
    new_state, stable = check_detection_stability(
        state, detected_type=1, now=100.0, min_position_frames=5
    )
    assert stable is False
    assert new_state.detected_type == 1
    assert new_state.position_frames == 4


def test_can_count_new_garbage():
    """New garbage type can always be counted."""
    state = DetectionState(
        detected_type=0,
        last_count_time=50.0,
    )
    can_count = can_count_new_garbage(
        state, detected_type=1, now=100.0, cooldown_ms=2000
    )
    assert can_count is True


def test_can_count_same_type_within_cooldown():
    """Same type within cooldown should not be counted."""
    state = DetectionState(
        detected_type=0,
        last_count_time=99.0,
    )
    can_count = can_count_new_garbage(
        state, detected_type=0, now=100.0, cooldown_ms=2000
    )
    # 100.0 - 99.0 = 1.0 seconds, which is < 2.0 seconds cooldown
    assert can_count is False


def test_can_count_same_type_after_cooldown():
    """Same type after cooldown can be counted."""
    state = DetectionState(
        detected_type=0,
        last_count_time=95.0,
    )
    can_count = can_count_new_garbage(
        state, detected_type=0, now=100.0, cooldown_ms=2000
    )
    # 100.0 - 95.0 = 5.0 seconds, which is >= 2.0 seconds cooldown
    assert can_count is True


def test_map_serial_payload():
    detection = {
        "class_id": 1,
        "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
    }

    class_id, x, y = map_serial_payload(detection, 640, 480)
    assert class_id == 1
    assert x == int(150 / 640 * 255)
    assert y == int(150 / 480 * 255)
