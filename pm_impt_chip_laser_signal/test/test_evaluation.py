import cv2
import numpy as np
import pytest

from pm_impt_chip_laser_signal.pm_impt_chip_laser_signal import (
    calculate_evaluation_value,
    calculate_temporal_median,
    evaluate_best_circle,
    stitch_evaluation_images,
)


def test_large_filled_circle_reaches_maximum_score():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (50, 50), 49, 255, thickness=cv2.FILLED)

    assert calculate_evaluation_value(image) == pytest.approx(100.0)


def test_larger_circle_scores_higher():
    small_circle = np.zeros((100, 100), dtype=np.uint8)
    large_circle = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(small_circle, (50, 50), 15, 255, thickness=cv2.FILLED)
    cv2.circle(large_circle, (50, 50), 35, 255, thickness=cv2.FILLED)

    assert calculate_evaluation_value(
        large_circle
    ) > calculate_evaluation_value(small_circle)


def test_smaller_circle_scores_higher_than_full_square():
    circle = np.zeros((100, 100), dtype=np.uint8)
    square = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(circle, (50, 50), 20, 255, thickness=cv2.FILLED)

    assert calculate_evaluation_value(circle) > calculate_evaluation_value(
        square
    )


def test_evaluation_is_zero_without_contours():
    image = np.zeros((100, 100), dtype=np.uint8)

    assert calculate_evaluation_value(image) == 0.0


def test_temporal_median_rejects_single_frame_spike():
    assert calculate_temporal_median(
        [100.0, 101.0, 500.0, 99.0, 102.0]
    ) == 101.0


def test_evaluation_image_contains_only_best_circle():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (20, 20), 8, 255, thickness=cv2.FILLED)
    cv2.circle(image, (70, 70), 18, 255, thickness=cv2.FILLED)

    _, selected_image = evaluate_best_circle(image)

    assert selected_image[20, 20] == 0
    assert selected_image[70, 70] == 255
    assert np.count_nonzero(selected_image) > 900


def test_dashboard_contains_both_images_and_dynamic_scale():
    camera_image = np.zeros((100, 120, 3), dtype=np.uint8)
    evaluation_image = np.zeros((100, 120), dtype=np.uint8)

    dashboard = stitch_evaluation_images(
        camera_image,
        evaluation_image,
        evaluation_value=50.0,
        evaluation_min=10.0,
        evaluation_max=100.0,
    )

    assert dashboard.shape[0] > camera_image.shape[0]
    assert dashboard.shape[1] > camera_image.shape[1] * 2
    assert dashboard.shape[2] == 3
