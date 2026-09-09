import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import time_ns

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from pm_skills_interfaces.msg import BoolStamped, Float64Stamped
from pm_vision_manager.va_py_modules.camera_ros_interfaces import (
    CameraRosInterfaces,
)
from pm_vision_manager.va_py_modules.image_processing_handler import (
    ImageProcessingHandler,
)
from pm_vision_manager.va_py_modules.vision_pipline_processing import (
    bgr2gray,
    roi,
    threshold,
    morphologyExOpening
)

CAMERA_FILE_NAME = 'pm_robot_basler_top_cam_1.yaml'
CAMERA_FILE_PATH = (
    '/home/pmlab/pm_Server/01_PM_Zelle/03_PM_DataBase/'
    'pm_assembly_database/Vision/vision_camera_configs'
)
VIDEO_OUTPUT_PATH = (
    '/home/pmlab/pm_Server/01_PM_Zelle/03_PM_DataBase/'
    'pm_assembly_database/RSAP_Processes/PhoenixD/'
    'IMPT_Chip_Fiber_Alignment/recordings'
)
CAMERA_TOPIC = 'Camera_Top_View/pylon_ros2_camera_node/image_raw'
WINDOW_NAME = 'Top camera'
INITIAL_WINDOW_WIDTH = 1000
FALLBACK_WINDOW_HEIGHT = 750
DEFAULT_EXPOSURE_PERCENT = 10.0
THRESHOLD_VALUE = 120
EX_OPEN_KERNEL_SIZE = 3 #7
VIDEO_FPS = 30.0
VIDEO_CODEC = 'mp4v'
CIRCLE_SCORE_THRESHOLD = 80.0
PANEL_HEADER_COLOR = (24, 28, 36)
INPUT_PANEL_COLOR = (255, 190, 40)
EVALUATION_PANEL_COLOR = (70, 220, 120)
TEMPORAL_MEDIAN_WINDOW = 5
CIRCULARITY_REFERENCE = 0.85
CIRCLE_FILL_REFERENCE = 0.90
SIZE_SCORE_EXPONENT = 0.50
CIRCULARITY_SCORE_EXPONENT = 2.0
CIRCLE_FILL_SCORE_EXPONENT = 2.0
MAX_CIRCLE_SCORE = 100.0
USE_TEMPORAL_MEDIAN_FILTER = False
STREAM_EVALUATION_MIN = 0.0


def calculate_temporal_median(evaluation_values):
    if not evaluation_values:
        return 0.0

    return float(np.median(evaluation_values))


def create_alignment_messages(evaluation_value, result_bool, image_header):
    """Create timestamped float and boolean values from one acquired image."""
    float_message = Float64Stamped()
    float_message.header = image_header
    float_message.data = evaluation_value
    bool_message = BoolStamped()
    bool_message.header = image_header
    bool_message.data = result_bool
    return float_message, bool_message


def get_header_stamp_ns(header):
    """Return the required non-zero image acquisition stamp."""
    stamp_ns = (
        int(header.stamp.sec) * 1_000_000_000
        + int(header.stamp.nanosec)
    )
    if stamp_ns <= 0:
        raise ValueError('Camera image has no acquisition timestamp.')
    return stamp_ns


def calculate_circle_score(contour, image_shape):
    contour_area = float(cv2.contourArea(contour))
    contour_perimeter = float(cv2.arcLength(contour, closed=True))
    if contour_area <= 0 or contour_perimeter <= 0:
        return 0.0

    _, enclosing_radius = cv2.minEnclosingCircle(contour)
    if enclosing_radius <= 0:
        return 0.0

    circularity = min(
        4.0 * np.pi * contour_area / contour_perimeter ** 2,
        1.0,
    )
    enclosing_circle_area = np.pi * enclosing_radius ** 2
    circle_fill = min(contour_area / enclosing_circle_area, 1.0)
    circularity_quality = min(
        circularity / CIRCULARITY_REFERENCE,
        1.0,
    )
    fill_quality = min(
        circle_fill / CIRCLE_FILL_REFERENCE,
        1.0,
    )

    image_height, image_width = image_shape[:2]
    maximum_radius = max(
        1.0,
        (min(image_height, image_width) - 2) / 2.0,
    )
    size_quality = min(enclosing_radius / maximum_radius, 1.0)
    return MAX_CIRCLE_SCORE * (
        size_quality ** SIZE_SCORE_EXPONENT
        * circularity_quality ** CIRCULARITY_SCORE_EXPONENT
        * fill_quality ** CIRCLE_FILL_SCORE_EXPONENT
    )


def evaluate_best_circle(binary_image):
    if binary_image.ndim != 2:
        raise ValueError(
            f'Expected a single-channel binary image, got '
            f'shape {binary_image.shape}'
        )

    selected_image = binary_image.copy()
    selected_image.fill(0)

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0, selected_image

    selected_contour = None
    evaluation_value = 0.0
    for contour in contours:
        contour_evaluation = calculate_circle_score(
            contour,
            binary_image.shape,
        )
        if selected_contour is None or contour_evaluation > evaluation_value:
            selected_contour = contour
            evaluation_value = contour_evaluation

    selected_mask = np.zeros_like(binary_image)
    cv2.drawContours(
        selected_mask,
        (selected_contour,),
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )
    selected_image = cv2.bitwise_and(
        binary_image,
        binary_image,
        mask=selected_mask,
    )
    return evaluation_value, selected_image


def evaluate_largest_contour(binary_image):
    return evaluate_best_circle(binary_image)


def calculate_evaluation_value(binary_image):
    evaluation_value, _ = evaluate_best_circle(binary_image)
    return evaluation_value


def calculate_white_pixel_distance_penalty(
    binary_image,
    image_processing_handler,
):
    """
    Return a 0..1 penalty for white pixels far from the image center.

    Distances are evaluated in coordinates of the initial camera image.  This
    is important because ``binary_image`` may contain only an ROI.  Squaring
    the normalized distance gives pixels near the center little influence,
    while pixels near the outer image boundary contribute a strong penalty.
    """
    if binary_image.ndim != 2:
        raise ValueError(
            f'Expected a single-channel binary image, got '
            f'shape {binary_image.shape}'
        )

    white_y_roi, white_x_roi = np.nonzero(binary_image)
    if white_x_roi.size == 0:
        return 0.0

    white_x_image, white_y_image = (
        image_processing_handler.CS_Conv_ROI_Pix_TO_Img_Pix(
            white_x_roi,
            white_y_roi,
        )
    )
    image_width = image_processing_handler.img_width
    image_height = image_processing_handler.img_height
    center_x = (image_width - 1) / 2.0
    center_y = (image_height - 1) / 2.0
    maximum_squared_distance = center_x ** 2 + center_y ** 2
    if maximum_squared_distance <= 0:
        return 0.0

    squared_distances = (
        (white_x_image - center_x) ** 2
        + (white_y_image - center_y) ** 2
    )
    return float(np.clip(
        np.mean(squared_distances / maximum_squared_distance),
        0.0,
        1.0,
    ))


def apply_white_pixel_distance_penalty(
    evaluation_value,
    binary_image,
    image_processing_handler,
):
    penalty = calculate_white_pixel_distance_penalty(
        binary_image,
        image_processing_handler,
    )
    return evaluation_value * (1.0 - penalty)


def add_video_panel_header(image, label, accent_color):
    image_height, image_width = image.shape[:2]
    header_height = max(36, round(image_height * 0.08))
    border_width = max(3, round(min(image_height, image_width) * 0.006))
    panel = cv2.copyMakeBorder(
        image,
        header_height,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        value=accent_color,
    )
    cv2.rectangle(
        panel,
        (border_width, 0),
        (panel.shape[1] - border_width - 1, header_height - 1),
        PANEL_HEADER_COLOR,
        thickness=-1,
    )
    accent_height = max(3, border_width)
    cv2.rectangle(
        panel,
        (border_width, header_height - accent_height),
        (panel.shape[1] - border_width - 1, header_height - 1),
        accent_color,
        thickness=-1,
    )

    font_scale = max(0.55, header_height / 58.0)
    font_thickness = max(1, round(font_scale * 2))
    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        font_thickness,
    )
    text_x = max(10, round(image_width * 0.025))
    text_y = max(
        text_size[1] + 4,
        (header_height + text_size[1]) // 2 - accent_height,
    )
    cv2.putText(
        panel,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (245, 248, 252),
        font_thickness,
        cv2.LINE_AA,
    )
    return panel


def format_evaluation_value(value):
    if abs(value) >= 1_000_000:
        return f'{value:.3e}'
    return f'{value:.2f}'


def get_viridis_color(fraction):
    """Return the OpenCV BGR Viridis color for a normalized value."""
    color_index = round(255 * float(np.clip(fraction, 0.0, 1.0)))
    color = cv2.applyColorMap(
        np.array([[color_index]], dtype=np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )[0, 0]
    return tuple(int(channel) for channel in color)


def get_recording_evaluation_range(evaluation_values):
    """Return the fixed 0-to-maximum scale used for finalized recordings."""
    if not evaluation_values:
        raise ValueError('No evaluation values are available')
    return STREAM_EVALUATION_MIN, max(
        STREAM_EVALUATION_MIN,
        max(evaluation_values),
    )


def add_evaluation_scale(
    image,
    evaluation_value,
    evaluation_min,
    evaluation_max,
):
    image_height, image_width = image.shape[:2]
    footer_height = max(72, round(image_height * 0.14))
    dashboard = cv2.copyMakeBorder(
        image,
        0,
        footer_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=PANEL_HEADER_COLOR,
    )

    padding = max(18, round(image_width * 0.018))
    track_left = round(image_width * 0.36)
    track_right = image_width - padding
    track_y = image_height + round(footer_height * 0.42)
    track_height = max(10, round(footer_height * 0.16))
    segment_count = 64
    for segment_index in range(segment_count):
        fraction = segment_index / (segment_count - 1)
        color = get_viridis_color(fraction)
        segment_left = round(
            track_left
            + (track_right - track_left) * segment_index / segment_count
        )
        segment_right = round(
            track_left
            + (track_right - track_left)
            * (segment_index + 1)
            / segment_count
        )
        cv2.rectangle(
            dashboard,
            (segment_left, track_y - track_height // 2),
            (segment_right, track_y + track_height // 2),
            color,
            thickness=-1,
        )

    value_range = evaluation_max - evaluation_min
    if value_range > 0:
        position = (evaluation_value - evaluation_min) / value_range
    else:
        position = 0.5
    position = max(0.0, min(position, 1.0))
    marker_x = round(track_left + position * (track_right - track_left))
    marker_radius = max(7, track_height)
    cv2.circle(
        dashboard,
        (marker_x, track_y),
        marker_radius,
        (245, 248, 252),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_DUPLEX
    label_scale = max(0.55, footer_height / 105.0)
    small_scale = max(0.42, footer_height / 145.0)
    cv2.putText(
        dashboard,
        f'SCORE  {format_evaluation_value(evaluation_value)}',
        (padding, image_height + round(footer_height * 0.48)),
        font,
        label_scale,
        (245, 248, 252),
        max(1, round(label_scale * 2)),
        cv2.LINE_AA,
    )
    label_y = image_height + round(footer_height * 0.84)
    cv2.putText(
        dashboard,
        f'MIN {format_evaluation_value(evaluation_min)}',
        (track_left, label_y),
        font,
        small_scale,
        (190, 198, 210),
        1,
        cv2.LINE_AA,
    )
    max_label = f'MAX {format_evaluation_value(evaluation_max)}'
    max_label_size, _ = cv2.getTextSize(
        max_label,
        font,
        small_scale,
        1,
    )
    cv2.putText(
        dashboard,
        max_label,
        (track_right - max_label_size[0], label_y),
        font,
        small_scale,
        (190, 198, 210),
        1,
        cv2.LINE_AA,
    )
    return dashboard


def stitch_evaluation_images(
    camera_image,
    evaluation_image,
    evaluation_value=None,
    evaluation_min=None,
    evaluation_max=None,
):
    if camera_image.ndim == 2:
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
    if evaluation_image.ndim == 2:
        evaluation_image = cv2.cvtColor(
            evaluation_image,
            cv2.COLOR_GRAY2BGR,
        )

    camera_height, camera_width = camera_image.shape[:2]
    if evaluation_image.shape[:2] != (camera_height, camera_width):
        evaluation_image = cv2.resize(
            evaluation_image,
            (camera_width, camera_height),
            interpolation=cv2.INTER_NEAREST,
        )

    input_panel = add_video_panel_header(
        camera_image,
        'INPUT  /  CAMERA',
        INPUT_PANEL_COLOR,
    )
    evaluation_panel = add_video_panel_header(
        evaluation_image,
        'FINAL  /  EVALUATION',
        EVALUATION_PANEL_COLOR,
    )
    image = cv2.hconcat((input_panel, evaluation_panel))
    if (
        evaluation_value is not None
        and evaluation_min is not None
        and evaluation_max is not None
    ):
        image = add_evaluation_scale(
            image,
            evaluation_value,
            evaluation_min,
            evaluation_max,
        )
    return image


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('impt_chip_laser_signal_alginment_checker')
        bool_output_topic_name = 'bool_alignment'
        float_output_topic_name = 'float_alignment'

        self.bool_output_topic = (
            f'{self.get_name()}/{bool_output_topic_name}'
        )

        self.float_output_topic = (
            f'{self.get_name()}/{float_output_topic_name}'
        )
        self.bool_publisher = self.create_publisher(
            BoolStamped,
            self.bool_output_topic,
            10,
        )
        self.float_publisher = self.create_publisher(
            Float64Stamped,
            self.float_output_topic,
            10,
        )
        # Treat all output topics as one recording trigger. Starting with no
        # known subscribers makes the first subscription on any topic the
        # transition that opens the window and starts recording.
        self.output_has_subscribers = False
        self.bool_output_has_subscribers = False
        self.float_output_has_subscribers = False
        self.subscription_monitor = self.create_timer(
            0.2,
            self.monitor_output_subscriptions,
        )
        self.logger = self.get_logger()
        self.bridge = CvBridge()
        self.latest_camera_image = None
        self.evaluation_image = None
        self.latest_evaluation_value = None
        self.latest_acquisition_ros_time_ns = None
        self.latest_image_receive_ros_time_ns = None
        self.evaluation_history = deque(maxlen=TEMPORAL_MEDIAN_WINDOW)
        self.image_lock = Lock()
        self.window_stop_event = Event()
        self.window_thread = None
        self.window_open = False
        self.result_bool = False

        self.declare_parameter('camera_topic', CAMERA_TOPIC)
        self.declare_parameter(
            'camera_config_path',
            str(Path(CAMERA_FILE_PATH) / CAMERA_FILE_NAME),
        )
        self.declare_parameter('area_threshold', CIRCLE_SCORE_THRESHOLD)
        self.score_threshold = self.get_parameter(
            'area_threshold'
        ).get_parameter_value().double_value
        self.declare_parameter(
            'use_temporal_median_filter',
            USE_TEMPORAL_MEDIAN_FILTER,
        )
        self.use_temporal_median_filter = self.get_parameter(
            'use_temporal_median_filter'
        ).get_parameter_value().bool_value

        self.ros_camera_interfaces = CameraRosInterfaces(self)
        self.image_processing_handler = ImageProcessingHandler(
            logger=self.logger,
            ros_camera_interfaces=self.ros_camera_interfaces,
        )

        camera_config_path = self.get_parameter(
            'camera_config_path'
        ).get_parameter_value().string_value
        self.image_processing_handler.set_camera_parameter_from_file(
            camera_config_path
        )
        # The camera helper performs only a short service-discovery check when
        # loading the configuration.  Retry here so normal DDS discovery
        # delays do not leave otherwise valid camera interfaces unavailable.
        self.wait_for_camera_interfaces()

        camera_topic = self.get_parameter(
            'camera_topic'
        ).get_parameter_value().string_value
        self.camera_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.logger.info(f'Subscribed to top camera topic: {camera_topic}')
        self.camera_exposure = DEFAULT_EXPOSURE_PERCENT

    def set_camera_exposure(self):
        self.wait_for_camera_interfaces()
        self.turn_off_lights()

        response = self.image_processing_handler.set_camera_exposure_time(
            self.camera_exposure
        )

        if not response.success:
            self.logger.error(
                'Could not set the top-camera exposure to '
                f'{self.camera_exposure:.1f}%.'
            )

    def wait_for_camera_interfaces(self):
        interfaces = (
            (
                'exposure',
                self.ros_camera_interfaces.exposure_time_interface,
                'client_exposure_time',
            ),
            (
                'coax light',
                self.ros_camera_interfaces.set_coax_light_bool_interface,
                'client_set_coax_light',
            ),
            (
                'ring light',
                self.ros_camera_interfaces.set_ring_light_interfaces,
                'client',
            ),
        )

        for interface_name, interface, client_attribute in interfaces:
            if interface.available:
                continue

            client = getattr(interface, client_attribute, None)
            if client is None:
                continue

            self.logger.info(
                f'Waiting for the camera {interface_name} service...'
            )
            if client.wait_for_service(timeout_sec=5.0):
                interface.available = True
                self.logger.info(
                    f'Camera {interface_name} interface is now available.'
                )
            else:
                self.logger.error(
                    f'Camera {interface_name} service is still unavailable.'
                )

    def turn_off_lights(self):
        self.image_processing_handler.disable_all_lights()
        self.logger.info('Requested all configured camera lights to turn off.')

    def camera_callback(self, message):
        try:
            image_receive_time_ns = self.get_clock().now().nanoseconds
            acquisition_time_ns = get_header_stamp_ns(message.header)
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            self.image_processing_handler.set_initial_image(image)
            self.image_processing_handler.init_begin()

            roi(
                image_processing_handler=self.image_processing_handler,
                ROI_center_x_c=0,
                ROI_center_y_c=0,
                ROI_height=30,
                ROI_width=30,
            )

            current_image = (
                self.image_processing_handler.get_processing_image()
            )

            bgr2gray(self.image_processing_handler)
            threshold(
                image_processing_handler=self.image_processing_handler,
                thresh=THRESHOLD_VALUE,
                maxval=255,
                type='THRESH_BINARY',
            )

            morphologyExOpening(
                image_processing_handler=self.image_processing_handler,
                kernelsize=EX_OPEN_KERNEL_SIZE
            )

            binary_image = self.image_processing_handler.get_processing_image()
            raw_evaluation_value, selected_image = evaluate_best_circle(
                binary_image
            )
            raw_evaluation_value = apply_white_pixel_distance_penalty(
                raw_evaluation_value,
                binary_image,
                self.image_processing_handler,
            )
            if self.use_temporal_median_filter:
                self.evaluation_history.append(raw_evaluation_value)
                evaluation_value = calculate_temporal_median(
                    self.evaluation_history
                )
            else:
                evaluation_value = raw_evaluation_value
            result_bool = evaluation_value > self.score_threshold

            with self.image_lock:
                self.latest_camera_image = current_image.copy()
                self.evaluation_image = selected_image
                self.latest_evaluation_value = evaluation_value
                self.result_bool = result_bool
                self.latest_acquisition_ros_time_ns = acquisition_time_ns
                self.latest_image_receive_ros_time_ns = image_receive_time_ns

            area_msg, result_msg = create_alignment_messages(
                evaluation_value,
                result_bool,
                message.header,
            )
            self.float_publisher.publish(area_msg)
            self.logger.info(f"Float_publisher: {evaluation_value}")
            self.bool_publisher.publish(result_msg)

            if self.float_output_has_subscribers:
                self.logger.info(
                    f'Publishing on {self.float_output_topic}: '
                    f'{area_msg.data} at '
                    f'{area_msg.header.stamp.sec}.'
                    f'{area_msg.header.stamp.nanosec:09d}'
                )
            if self.bool_output_has_subscribers:
                self.logger.info(
                    f'Publishing on {self.bool_output_topic}: '
                    f'{result_msg.data}'
                )

        except Exception as error:
            self.logger.error(
                'Could not process the top-camera image '
                f'(encoding={message.encoding}, size='
                f'{message.width}x{message.height}): '
                f'{type(error).__name__}: {error}'
            )

    def monitor_output_subscriptions(self):
        self.bool_output_has_subscribers = (
            self.bool_publisher.get_subscription_count() > 0
        )
        self.float_output_has_subscribers = (
            self.float_publisher.get_subscription_count() > 0
        )
        subscribed_topics = [
            topic
            for topic, has_subscribers in (
                (
                    self.bool_output_topic,
                    self.bool_output_has_subscribers,
                ),
                (
                    self.float_output_topic,
                    self.float_output_has_subscribers,
                ),
            )
            if has_subscribers
        ]
        has_subscribers = bool(subscribed_topics)

        if has_subscribers:
            if not self.output_has_subscribers:
                topic_list = ', '.join(subscribed_topics)
                self.logger.info(
                    f'Subscriber detected on {topic_list}; '
                    'opening the camera window.'
                )
                Thread(target=self.set_camera_exposure, daemon=True).start()

            if (
                self.window_thread is None
                or not self.window_thread.is_alive()
            ):
                self.start_window_thread()

        elif not has_subscribers and self.output_has_subscribers:
            self.logger.info(
                'No subscribers remain on any output topic; '
                'closing the camera window.'
            )
            self.stop_window_thread()

        self.output_has_subscribers = has_subscribers

    def start_window_thread(self):
        if self.window_thread is not None and self.window_thread.is_alive():
            return

        self.window_stop_event.clear()
        self.window_thread = Thread(
            target=self.display_camera_window,
            daemon=True,
        )
        self.window_thread.start()

    def stop_window_thread(self):
        self.window_stop_event.set()

    def display_camera_window(self):
        video_writer = None
        video_path = None
        source_video_path = None
        timestamp_file = None
        timestamp_writer = None
        timestamp_path = None
        frame_index = 0
        evaluation_min = STREAM_EVALUATION_MIN
        evaluation_max = None
        recorded_evaluation_values = []

        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            with self.image_lock:
                camera_image = (
                    None if self.latest_camera_image is None
                    else self.latest_camera_image.copy()
                )
                evaluation_image = (
                    None if self.evaluation_image is None
                    else self.evaluation_image.copy()
                )
                evaluation_value = self.latest_evaluation_value

            image = None
            if (
                camera_image is not None
                and evaluation_image is not None
                and evaluation_value is not None
            ):
                evaluation_max = max(evaluation_min, evaluation_value)
                base_image = stitch_evaluation_images(
                    camera_image,
                    evaluation_image,
                )
                image = add_evaluation_scale(
                    base_image,
                    evaluation_value,
                    evaluation_min,
                    evaluation_max,
                )

            if image is None:
                initial_window_height = FALLBACK_WINDOW_HEIGHT
            else:
                image_height, image_width = image.shape[:2]
                initial_window_height = round(
                    INITIAL_WINDOW_WIDTH * image_height / image_width
                )

            cv2.resizeWindow(
                WINDOW_NAME,
                INITIAL_WINDOW_WIDTH,
                initial_window_height,
            )
            self.window_open = True

            while not self.window_stop_event.wait(1.0 / VIDEO_FPS):
                with self.image_lock:
                    camera_image = (
                        None if self.latest_camera_image is None
                        else self.latest_camera_image.copy()
                    )
                    evaluation_image = (
                        None if self.evaluation_image is None
                        else self.evaluation_image.copy()
                    )
                    evaluation_value = self.latest_evaluation_value
                    acquisition_time_ns = self.latest_acquisition_ros_time_ns
                    image_receive_time_ns = (
                        self.latest_image_receive_ros_time_ns
                    )

                image = None
                if (
                    camera_image is not None
                    and evaluation_image is not None
                    and evaluation_value is not None
                ):
                    if evaluation_max is None:
                        evaluation_max = max(
                            evaluation_min,
                            evaluation_value,
                        )
                    else:
                        evaluation_max = max(
                            evaluation_max,
                            evaluation_value,
                        )
                    base_image = stitch_evaluation_images(
                        camera_image,
                        evaluation_image,
                    )
                    image = add_evaluation_scale(
                        base_image,
                        evaluation_value,
                        evaluation_min,
                        evaluation_max,
                    )

                if image is not None:
                    if video_writer is None:
                        video_path = self.create_video_path()
                        source_video_path = video_path.with_name(
                            f'{video_path.stem}_source{video_path.suffix}'
                        )
                        _, video_writer = self.create_video_writer(
                            base_image,
                            source_video_path,
                        )
                        timestamp_path = video_path.with_suffix('.csv')
                        timestamp_file = timestamp_path.open(
                            'w',
                            encoding='utf-8',
                            newline='',
                        )
                        timestamp_writer = csv.writer(timestamp_file)
                        timestamp_writer.writerow((
                            'frame_index',
                            'ros_time_sec',
                            'ros_time_nanosec',
                            'ros_time_ns',
                            'image_acquisition_ros_time_sec',
                            'image_acquisition_ros_time_nanosec',
                            'image_acquisition_ros_time_ns',
                            'image_receive_ros_time_ns',
                            'acquisition_to_image_receive_ms',
                            'computer_time_unix_ns',
                            'computer_time_iso8601',
                            'evaluation_value',
                            'stream_evaluation_min',
                            'stream_evaluation_max',
                        ))
                        self.logger.info(
                            f'Recording frame timestamps to {timestamp_path}'
                        )

                    ros_time_ns = self.get_clock().now().nanoseconds
                    ros_time_sec, ros_time_nanosec = divmod(
                        ros_time_ns,
                        1_000_000_000,
                    )
                    acquisition_time_sec, acquisition_time_nanosec = divmod(
                        acquisition_time_ns,
                        1_000_000_000,
                    )
                    computer_time_ns = time_ns()
                    computer_time_sec, computer_time_nanosec = divmod(
                        computer_time_ns,
                        1_000_000_000,
                    )
                    computer_time_iso8601 = datetime.fromtimestamp(
                        computer_time_sec
                    ).astimezone().replace(
                        microsecond=computer_time_nanosec // 1_000,
                    ).isoformat(timespec='microseconds')

                    cv2.imshow(WINDOW_NAME, image)
                    video_writer.write(base_image)
                    recorded_evaluation_values.append(evaluation_value)
                    timestamp_writer.writerow((
                        frame_index,
                        ros_time_sec,
                        ros_time_nanosec,
                        ros_time_ns,
                        acquisition_time_sec,
                        acquisition_time_nanosec,
                        acquisition_time_ns,
                        image_receive_time_ns,
                        (image_receive_time_ns - acquisition_time_ns) / 1e6,
                        computer_time_ns,
                        computer_time_iso8601,
                        evaluation_value,
                        evaluation_min,
                        evaluation_max,
                    ))
                    frame_index += 1
                    if frame_index % round(VIDEO_FPS) == 0:
                        timestamp_file.flush()
                cv2.waitKey(1)
        except Exception as error:
            self.logger.error(
                f'Could not display the top-camera window: {error}'
            )
        finally:
            # Close the GUI as soon as streaming stops.  Rendering the final
            # fixed-scale video can take considerably longer and must not keep
            # the live window visible after the last subscriber disconnects.
            if self.window_open:
                try:
                    cv2.destroyWindow(WINDOW_NAME)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
                self.window_open = False

            if timestamp_file is not None:
                timestamp_file.close()
                self.logger.info(
                    f'Saved frame timestamps to {timestamp_path}'
                )

            if video_writer is not None:
                video_writer.release()
                try:
                    self.render_final_video(
                        source_video_path,
                        video_path,
                        recorded_evaluation_values,
                    )
                    source_video_path.unlink()
                    self.logger.info(
                        f'Saved camera recording to {video_path}'
                    )
                except Exception as error:
                    self.logger.error(
                        'Could not render the fixed-scale recording: '
                        f'{error}. Source video retained at '
                        f'{source_video_path}'
                    )

    def create_video_path(self):
        recordings_directory = Path(VIDEO_OUTPUT_PATH)
        recordings_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return recordings_directory / f'top_camera_{timestamp}.mp4'

    def create_video_writer(self, image, video_path=None):
        if video_path is None:
            video_path = self.create_video_path()

        image_height, image_width = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            VIDEO_FPS,
            (image_width, image_height),
        )

        if not video_writer.isOpened():
            video_writer.release()
            raise RuntimeError(f'Could not create video file {video_path}')

        self.logger.info(f'Recording camera video to {video_path}')
        return video_path, video_writer

    def render_final_video(
        self,
        source_video_path,
        video_path,
        evaluation_values,
    ):
        if not evaluation_values:
            raise RuntimeError('No recorded frames are available to render')

        evaluation_min, evaluation_max = get_recording_evaluation_range(
            evaluation_values
        )
        max_frame_index = evaluation_values.index(evaluation_max)
        self.logger.info(
            'Rendering final video with fixed evaluation range '
            f'[{evaluation_min}, {evaluation_max}]'
        )

        video_capture = cv2.VideoCapture(str(source_video_path))
        if not video_capture.isOpened():
            raise RuntimeError(
                f'Could not open source video {source_video_path}'
            )

        final_writer = None
        rendered_frame_count = 0
        max_frame = None
        try:
            while rendered_frame_count < len(evaluation_values):
                frame_available, frame = video_capture.read()
                if not frame_available:
                    break

                final_frame = add_evaluation_scale(
                    frame,
                    evaluation_values[rendered_frame_count],
                    evaluation_min,
                    evaluation_max,
                )
                if final_writer is None:
                    _, final_writer = self.create_video_writer(
                        final_frame,
                        video_path,
                    )
                final_writer.write(final_frame)
                if rendered_frame_count == max_frame_index:
                    max_frame = final_frame.copy()
                rendered_frame_count += 1
        finally:
            video_capture.release()
            if final_writer is not None:
                final_writer.release()

        if rendered_frame_count != len(evaluation_values):
            raise RuntimeError(
                f'Rendered {rendered_frame_count} of '
                f'{len(evaluation_values)} recorded frames'
            )

        max_image_path = video_path.with_name(
            f'{video_path.stem}_max.png'
        )
        image_saved = (
            max_frame is not None
            and cv2.imwrite(str(max_image_path), max_frame)
        )
        if not image_saved:
            raise RuntimeError(
                f'Could not save maximum-value image to {max_image_path}'
            )
        self.logger.info(
            'Saved maximum-value frame '
            f'(value={evaluation_max}) to {max_image_path}'
        )

    def destroy_node(self):
        self.stop_window_thread()
        if self.window_thread is not None:
            self.window_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()
    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
