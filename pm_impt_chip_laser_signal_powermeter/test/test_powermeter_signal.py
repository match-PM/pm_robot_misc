import cv2
import numpy as np
import pytest
from builtin_interfaces.msg import Time

from pm_impt_chip_laser_signal_powermeter.pm_impt_chip_laser_signal_powermeter import (  # noqa: E501
    calculate_value_range,
    create_stamped_power_message,
    crop_percent_roi,
    format_power,
    render_value_scale,
    ros_stamp_to_ns,
)


def test_stamped_message_preserves_supplied_receipt_time():
    stamp = Time(sec=123, nanosec=456)

    message = create_stamped_power_message(2.5e-3, stamp)

    assert message.header.stamp == stamp
    assert message.header.frame_id == 'thorlabs_pm100usb'
    assert message.data == 2.5e-3
    assert ros_stamp_to_ns(stamp) == 123_000_000_456


def test_value_range_uses_observed_minimum_and_maximum():
    assert calculate_value_range([3e-6, 1e-6, 2e-6]) == (1e-6, 3e-6)
    with pytest.raises(ValueError, match='No finite'):
        calculate_value_range([float('nan')])


def test_camera_roi_matches_centered_thirty_percent_settings():
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

    cropped = crop_percent_roi(image)

    assert cropped.shape == (30, 60, 3)
    assert np.array_equal(cropped, image[35:65, 70:130])


def test_power_format_uses_si_prefix():
    assert format_power(0.002) == '2 mW'
    assert format_power(3e-6) == '3 uW'


def test_one_dimensional_scale_renders_marker_and_gradient():
    image = render_value_scale(2.0, 1.0, 3.0)

    assert image.shape == (260, 1100, 3)
    assert image.dtype == np.uint8
    assert np.unique(image[135], axis=0).shape[0] > 100
    assert np.any(image == cv2.applyColorMap(
        np.array([[255]], dtype=np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )[0, 0])
