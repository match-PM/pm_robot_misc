import cv2
import numpy as np
import pytest

from pm_impt_chip_laser_signal.pm_impt_chip_laser_signal import (
    apply_white_pixel_distance_penalty,
    calculate_evaluation_value,
    calculate_temporal_median,
    calculate_white_pixel_distance_penalty,
    create_alignment_messages,
    evaluate_best_circle,
    get_header_stamp_ns,
    get_recording_evaluation_range,
    get_viridis_color,
    stitch_evaluation_images,
)
from std_msgs.msg import Header


class RoiCoordinateHandler:

    def __init__(self, image_shape, roi_offset=(0, 0)):
        self.img_height, self.img_width = image_shape
        self.roi_offset = roi_offset

    def CS_Conv_ROI_Pix_TO_Img_Pix(self, x_roi, y_roi):
        return (
            x_roi + self.roi_offset[0],
            y_roi + self.roi_offset[1],
        )


def test_alignment_message_preserves_image_acquisition_stamp():
    header = Header()
    header.stamp.sec = 123
    header.stamp.nanosec = 456
    header.frame_id = 'camera'

    float_message, bool_message = create_alignment_messages(
        42.5,
        True,
        header,
    )

    assert float_message.header.stamp.sec == 123
    assert float_message.header.stamp.nanosec == 456
    assert float_message.header.frame_id == 'camera'
    assert float_message.data == 42.5
    assert bool_message.header.stamp.sec == 123
    assert bool_message.header.stamp.nanosec == 456
    assert bool_message.header.frame_id == 'camera'
    assert bool_message.data is True
    assert get_header_stamp_ns(header) == 123_000_000_456

    header.stamp.sec = 0
    header.stamp.nanosec = 0
    with pytest.raises(ValueError, match='no acquisition timestamp'):
        get_header_stamp_ns(header)


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


def test_white_pixels_far_from_full_image_center_receive_larger_penalty():
    centered = np.zeros((40, 40), dtype=np.uint8)
    far_from_center = np.zeros((40, 40), dtype=np.uint8)
    centered[15:25, 15:25] = 255
    far_from_center[0:10, 0:10] = 255
    handler = RoiCoordinateHandler((100, 100), roi_offset=(30, 30))

    centered_penalty = calculate_white_pixel_distance_penalty(
        centered,
        handler,
    )
    far_penalty = calculate_white_pixel_distance_penalty(
        far_from_center,
        handler,
    )

    assert far_penalty > centered_penalty
    assert apply_white_pixel_distance_penalty(
        80.0,
        far_from_center,
        handler,
    ) < apply_white_pixel_distance_penalty(
        80.0,
        centered,
        handler,
    )


def test_white_pixel_penalty_uses_roi_offset():
    white_pixels = np.full((10, 10), 255, dtype=np.uint8)
    centered_roi = RoiCoordinateHandler((100, 100), roi_offset=(45, 45))
    corner_roi = RoiCoordinateHandler((100, 100), roi_offset=(0, 0))

    assert calculate_white_pixel_distance_penalty(
        white_pixels,
        corner_roi,
    ) > calculate_white_pixel_distance_penalty(
        white_pixels,
        centered_roi,
    )


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


def test_evaluation_scale_uses_viridis_endpoints():
    expected = cv2.applyColorMap(
        np.array([[0, 255]], dtype=np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )[0]

    assert get_viridis_color(0.0) == tuple(int(value) for value in expected[0])
    assert get_viridis_color(1.0) == tuple(int(value) for value in expected[1])


def test_final_recording_scale_starts_at_zero_and_uses_recorded_maximum():
    assert get_recording_evaluation_range([8.0, 12.5, 10.0]) == (
        0.0,
        12.5,
    )
