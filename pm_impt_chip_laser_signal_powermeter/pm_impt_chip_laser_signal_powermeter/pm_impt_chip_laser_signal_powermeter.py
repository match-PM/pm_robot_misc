"""Timestamp, display, and record the IMPT chip laser power signal."""

import csv
import math
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
from std_msgs.msg import Float64

from pm_skills_interfaces.msg import Float64Stamped


DEFAULT_INPUT_TOPIC = 'thorlabs_pm100usb/power'
DEFAULT_CAMERA_TOPIC = 'Camera_Top_View/pylon_ros2_camera_node/image_raw'
DEFAULT_OUTPUT_DIRECTORY = (
    '/home/pmlab/pm_Server/01_PM_Zelle/03_PM_DataBase/'
    'pm_assembly_database/RSAP_Processes/PhoenixD/'
    'IMPT_Chip_Fiber_Alignment/recordings'
)
WINDOW_NAME = 'IMPT chip laser power'
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 260
BACKGROUND_COLOR = (24, 28, 36)
TEXT_COLOR = (245, 248, 252)
VIDEO_CODEC = 'mp4v'
DEFAULT_CAMERA_VIDEO_FPS = 30.0
ROI_CENTER_X_PERCENT = 0.0
ROI_CENTER_Y_PERCENT = 0.0
ROI_HEIGHT_PERCENT = 30.0
ROI_WIDTH_PERCENT = 30.0


def create_stamped_power_message(value, stamp):
    """Create a stamped power message using the supplied ROS timestamp."""
    message = Float64Stamped()
    message.header.stamp = stamp
    message.header.frame_id = 'thorlabs_pm100usb'
    message.data = float(value)
    return message


def ros_stamp_to_ns(stamp):
    """Convert a ROS time message to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def crop_percent_roi(
    image,
    center_x_percent=ROI_CENTER_X_PERCENT,
    center_y_percent=ROI_CENTER_Y_PERCENT,
    height_percent=ROI_HEIGHT_PERCENT,
    width_percent=ROI_WIDTH_PERCENT,
):
    """Apply the percentage ROI used by the camera signal evaluator."""
    if image.ndim not in (2, 3):
        raise ValueError(f'Unsupported camera image shape: {image.shape}')
    if not 0.0 < height_percent <= 100.0:
        raise ValueError('ROI height must be in the range (0, 100]')
    if not 0.0 < width_percent <= 100.0:
        raise ValueError('ROI width must be in the range (0, 100]')

    image_height, image_width = image.shape[:2]
    half_height = int(image_height * height_percent / 200.0)
    half_width = int(image_width * width_percent / 200.0)
    center_x = int(image_width / 2 + image_width * center_x_percent / 200.0)
    center_y = int(
        image_height / 2 - image_height * center_y_percent / 200.0
    )
    top = center_y - half_height
    bottom = center_y + half_height
    left = center_x - half_width
    right = center_x + half_width
    if top < 0 or left < 0 or bottom > image_height or right > image_width:
        raise ValueError('The configured ROI extends outside the camera image')
    if top == bottom or left == right:
        raise ValueError('The configured ROI is empty for this camera image')
    return image[top:bottom, left:right].copy()


def calculate_value_range(values):
    """Return the observed minimum and maximum finite values."""
    finite_values = [float(value) for value in values if math.isfinite(value)]
    if not finite_values:
        raise ValueError('No finite power values are available')
    return min(finite_values), max(finite_values)


def format_power(value):
    """Format a power in watts with a readable SI prefix."""
    absolute_value = abs(value)
    for scale, suffix in (
        (1.0, 'W'),
        (1e-3, 'mW'),
        (1e-6, 'uW'),
        (1e-9, 'nW'),
        (1e-12, 'pW'),
    ):
        if absolute_value >= scale or scale == 1e-12:
            return f'{value / scale:.4g} {suffix}'
    return f'{value:.4g} W'


def render_value_scale(value, minimum, maximum):
    """Render a horizontal one-dimensional scale and current-value marker."""
    image = np.full(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    track_left = 70
    track_right = WINDOW_WIDTH - 70
    track_y = 135
    track_height = 28
    for index in range(256):
        color = cv2.applyColorMap(
            np.array([[index]], dtype=np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )[0, 0]
        left = round(track_left + index * (track_right - track_left) / 256)
        right = round(
            track_left + (index + 1) * (track_right - track_left) / 256
        )
        cv2.rectangle(
            image,
            (left, track_y - track_height // 2),
            (right, track_y + track_height // 2),
            tuple(int(channel) for channel in color),
            thickness=-1,
        )

    fraction = 0.5 if maximum == minimum else (
        (value - minimum) / (maximum - minimum)
    )
    marker_x = round(
        track_left + float(np.clip(fraction, 0.0, 1.0))
        * (track_right - track_left)
    )
    cv2.circle(
        image,
        (marker_x, track_y),
        23,
        TEXT_COLOR,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f'POWER  {format_power(value)}',
        (track_left, 65),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f'MIN {format_power(minimum)}',
        (track_left, 205),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (190, 198, 210),
        1,
        cv2.LINE_AA,
    )
    maximum_label = f'MAX {format_power(maximum)}'
    label_size, _ = cv2.getTextSize(
        maximum_label,
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        1,
    )
    cv2.putText(
        image,
        maximum_label,
        (track_right - label_size[0], 205),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (190, 198, 210),
        1,
        cv2.LINE_AA,
    )
    return image


class PowerMeterSignalNode(Node):
    """Republish and record timestamped PM100USB measurements."""

    def __init__(self):
        super().__init__('pm_impt_chip_laser_signal_powermeter')
        self.declare_parameter('input_topic', DEFAULT_INPUT_TOPIC)
        self.declare_parameter('camera_topic', DEFAULT_CAMERA_TOPIC)
        self.declare_parameter('output_topic', 'float_alignment')
        self.declare_parameter('output_directory', DEFAULT_OUTPUT_DIRECTORY)
        self.declare_parameter(
            'camera_video_fps',
            DEFAULT_CAMERA_VIDEO_FPS,
        )
        self.declare_parameter('roi_center_x', ROI_CENTER_X_PERCENT)
        self.declare_parameter('roi_center_y', ROI_CENTER_Y_PERCENT)
        self.declare_parameter('roi_height', ROI_HEIGHT_PERCENT)
        self.declare_parameter('roi_width', ROI_WIDTH_PERCENT)

        input_topic = self.get_parameter('input_topic').value
        camera_topic = self.get_parameter('camera_topic').value
        output_topic_name = self.get_parameter('output_topic').value
        self.camera_video_fps = self.get_parameter('camera_video_fps').value
        if self.camera_video_fps <= 0.0:
            raise ValueError('camera_video_fps must be positive')
        self.roi_center_x = self.get_parameter('roi_center_x').value
        self.roi_center_y = self.get_parameter('roi_center_y').value
        self.roi_height = self.get_parameter('roi_height').value
        self.roi_width = self.get_parameter('roi_width').value
        self.output_directory = Path(
            self.get_parameter('output_directory').value
        )
        self.output_topic = f'{self.get_name()}/{output_topic_name}'

        self.publisher = self.create_publisher(
            Float64Stamped,
            self.output_topic,
            10,
        )
        self.subscription = self.create_subscription(
            Float64,
            input_topic,
            self.power_callback,
            10,
        )
        self.camera_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.monitor_timer = self.create_timer(
            0.2,
            self.monitor_output_subscriptions,
        )

        self.data_lock = Lock()
        self.bridge = CvBridge()
        self.recording = False
        self.latest_value = None
        self.session_rows = []
        self.camera_rows = []
        self.session_started_ns = None
        self.session_output_stem = None
        self.camera_video_writer = None
        self.camera_recording_failed = False
        self.latest_camera_roi = None
        self.latest_camera_receipt_ns = None
        self.maximum_power_value = None
        self.maximum_power_time_ns = None
        self.maximum_power_camera_image = None
        self.maximum_power_camera_distance_ns = None
        self.window_stop_event = Event()
        self.window_thread = None

        self.get_logger().info(f'Subscribed to power topic: {input_topic}')
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')
        self.get_logger().info(
            f'Publishing timestamped power on: {self.output_topic}'
        )

    def power_callback(self, source_message):
        value = float(source_message.data)
        if not math.isfinite(value):
            self.get_logger().warning(
                f'Ignoring non-finite power measurement: {value}'
            )
            return

        receipt_time = self.get_clock().now()
        stamped_message = create_stamped_power_message(
            value,
            receipt_time.to_msg(),
        )
        self.publisher.publish(stamped_message)

        ros_time_ns = receipt_time.nanoseconds
        computer_time_ns = time_ns()
        with self.data_lock:
            self.latest_value = value
            if self.recording:
                if self.session_started_ns is None:
                    self.session_started_ns = ros_time_ns
                self.session_rows.append((
                    len(self.session_rows),
                    ros_time_ns,
                    computer_time_ns,
                    value,
                ))
                if (
                    self.maximum_power_value is None
                    or value > self.maximum_power_value
                ):
                    self.maximum_power_value = value
                    self.maximum_power_time_ns = ros_time_ns
                    self.maximum_power_camera_image = None
                    self.maximum_power_camera_distance_ns = None
                    if self.latest_camera_roi is not None:
                        self.maximum_power_camera_image = (
                            self.latest_camera_roi.copy()
                        )
                        self.maximum_power_camera_distance_ns = abs(
                            self.latest_camera_receipt_ns - ros_time_ns
                        )

    def camera_callback(self, message):
        """Record the unprocessed camera stream during an active session."""
        with self.data_lock:
            should_record = self.recording and not self.camera_recording_failed
        if not should_record:
            return

        try:
            receipt_ros_time_ns = self.get_clock().now().nanoseconds
            computer_time_ns = time_ns()
            acquisition_ros_time_ns = ros_stamp_to_ns(message.header.stamp)
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
            image = crop_percent_roi(
                image,
                center_x_percent=self.roi_center_x,
                center_y_percent=self.roi_center_y,
                height_percent=self.roi_height,
                width_percent=self.roi_width,
            )
            with self.data_lock:
                if not self.recording:
                    return
                if self.camera_video_writer is None:
                    self.camera_video_writer = self.create_camera_video_writer(
                        image,
                    )
                self.camera_video_writer.write(image)
                self.camera_rows.append((
                    len(self.camera_rows),
                    acquisition_ros_time_ns,
                    receipt_ros_time_ns,
                    computer_time_ns,
                ))
                self.latest_camera_roi = image.copy()
                self.latest_camera_receipt_ns = receipt_ros_time_ns
                if self.maximum_power_time_ns is not None:
                    distance_ns = abs(
                        receipt_ros_time_ns - self.maximum_power_time_ns
                    )
                    if (
                        self.maximum_power_camera_distance_ns is None
                        or distance_ns < self.maximum_power_camera_distance_ns
                    ):
                        self.maximum_power_camera_image = image.copy()
                        self.maximum_power_camera_distance_ns = distance_ns
        except Exception as error:
            with self.data_lock:
                self.camera_recording_failed = True
            self.get_logger().error(
                'Could not record the camera stream: '
                f'{type(error).__name__}: {error}'
            )

    def monitor_output_subscriptions(self):
        has_subscribers = self.publisher.get_subscription_count() > 0
        if has_subscribers and not self.recording:
            self.start_recording()
        elif not has_subscribers and self.recording:
            self.stop_recording()

    def start_recording(self):
        output_stem = self.create_output_stem()
        with self.data_lock:
            self.session_rows = []
            self.camera_rows = []
            self.session_started_ns = None
            self.session_output_stem = output_stem
            self.camera_video_writer = None
            self.camera_recording_failed = False
            self.latest_camera_roi = None
            self.latest_camera_receipt_ns = None
            self.maximum_power_value = None
            self.maximum_power_time_ns = None
            self.maximum_power_camera_image = None
            self.maximum_power_camera_distance_ns = None
            self.recording = True
        self.get_logger().info(
            f'Subscriber detected on {self.output_topic}; '
            'starting power recording.'
        )
        self.window_stop_event.clear()
        self.window_thread = Thread(target=self.display_window, daemon=True)
        self.window_thread.start()

    def stop_recording(self):
        with self.data_lock:
            self.recording = False
            rows = list(self.session_rows)
            camera_rows = list(self.camera_rows)
            output_stem = self.session_output_stem
            camera_video_writer = self.camera_video_writer
            maximum_power_value = self.maximum_power_value
            maximum_power_camera_image = self.maximum_power_camera_image
            self.session_rows = []
            self.camera_rows = []
            self.session_started_ns = None
            self.session_output_stem = None
            self.camera_video_writer = None
            self.latest_camera_roi = None
            self.latest_camera_receipt_ns = None
            self.maximum_power_value = None
            self.maximum_power_time_ns = None
            self.maximum_power_camera_image = None
            self.maximum_power_camera_distance_ns = None
        if camera_video_writer is not None:
            camera_video_writer.release()
        self.window_stop_event.set()
        if self.window_thread is not None:
            self.window_thread.join()
            self.window_thread = None
        if rows:
            try:
                self.save_recording(rows, output_stem)
            except Exception as error:
                self.get_logger().error(
                    f'Could not save the power recording: {error}'
                )
        else:
            self.get_logger().warning(
                'Recording stopped before a power value was received.'
            )
        if camera_rows:
            try:
                self.save_camera_timestamps(camera_rows, output_stem)
            except Exception as error:
                self.get_logger().error(
                    f'Could not save camera timestamps: {error}'
                )
        else:
            self.get_logger().warning(
                'Recording stopped before a camera image was received.'
            )
        if maximum_power_camera_image is not None:
            try:
                self.save_maximum_power_image(
                    maximum_power_camera_image,
                    maximum_power_value,
                    output_stem,
                )
            except Exception as error:
                self.get_logger().error(
                    f'Could not save maximum-power camera image: {error}'
                )

    def display_window(self):
        window_open = False
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
            window_open = True
            while not self.window_stop_event.wait(1.0 / 30.0):
                with self.data_lock:
                    values = [row[3] for row in self.session_rows]
                    latest_value = self.latest_value
                if latest_value is not None:
                    if values:
                        minimum, maximum = calculate_value_range(values)
                    else:
                        minimum = maximum = latest_value
                    cv2.imshow(
                        WINDOW_NAME,
                        render_value_scale(latest_value, minimum, maximum),
                    )
                cv2.waitKey(1)
        except Exception as error:
            self.get_logger().error(
                f'Could not display the power scale: {error}'
            )
        finally:
            if window_open:
                try:
                    cv2.destroyWindow(WINDOW_NAME)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

    def create_output_stem(self):
        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return self.output_directory / f'powermeter_{timestamp}'

    def create_camera_video_writer(self, image):
        video_path = self.session_output_stem.with_name(
            f'{self.session_output_stem.name}_camera.mp4'
        )
        image_height, image_width = image.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
            self.camera_video_fps,
            (image_width, image_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f'Could not create camera video {video_path}')
        self.get_logger().info(f'Recording camera stream to {video_path}')
        return writer

    def save_recording(self, rows, output_stem):
        csv_path = output_stem.with_suffix('.csv')
        plot_path = output_stem.with_suffix('.png')
        values = [row[3] for row in rows]
        minimum, maximum = calculate_value_range(values)
        start_ros_time_ns = rows[0][1]

        with csv_path.open('w', encoding='utf-8', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow((
                'sample_index',
                'elapsed_time_s',
                'ros_time_sec',
                'ros_time_nanosec',
                'ros_time_ns',
                'computer_time_unix_ns',
                'computer_time_iso8601',
                'power_w',
                'recording_min_w',
                'recording_max_w',
            ))
            for sample_index, ros_time_ns, computer_time_ns, value in rows:
                ros_sec, ros_nanosec = divmod(ros_time_ns, 1_000_000_000)
                computer_sec, computer_nanosec = divmod(
                    computer_time_ns,
                    1_000_000_000,
                )
                computer_time = datetime.fromtimestamp(
                    computer_sec,
                ).astimezone().replace(
                    microsecond=computer_nanosec // 1_000,
                ).isoformat(timespec='microseconds')
                writer.writerow((
                    sample_index,
                    (ros_time_ns - start_ros_time_ns) / 1e9,
                    ros_sec,
                    ros_nanosec,
                    ros_time_ns,
                    computer_time_ns,
                    computer_time,
                    value,
                    minimum,
                    maximum,
                ))

        self.save_plot(rows, plot_path, minimum, maximum)
        self.get_logger().info(
            f'Saved {len(rows)} power samples to {csv_path}'
        )
        self.get_logger().info(
            f'Saved power plot (min={minimum:g} W, max={maximum:g} W) '
            f'to {plot_path}'
        )

    def save_camera_timestamps(self, rows, output_stem):
        """Save camera acquisition and receipt times alongside the video."""
        timestamp_path = output_stem.with_name(
            f'{output_stem.name}_camera_timestamps.csv'
        )
        first_receipt_ns = rows[0][2]
        with timestamp_path.open(
            'w',
            encoding='utf-8',
            newline='',
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow((
                'frame_index',
                'elapsed_time_s',
                'image_acquisition_ros_time_sec',
                'image_acquisition_ros_time_nanosec',
                'image_acquisition_ros_time_ns',
                'image_receive_ros_time_sec',
                'image_receive_ros_time_nanosec',
                'image_receive_ros_time_ns',
                'computer_time_unix_ns',
                'computer_time_iso8601',
            ))
            for (
                frame_index,
                acquisition_ns,
                receipt_ns,
                computer_ns,
            ) in rows:
                acquisition_sec, acquisition_nanosec = divmod(
                    acquisition_ns,
                    1_000_000_000,
                )
                receipt_sec, receipt_nanosec = divmod(
                    receipt_ns,
                    1_000_000_000,
                )
                computer_sec, computer_nanosec = divmod(
                    computer_ns,
                    1_000_000_000,
                )
                computer_time = datetime.fromtimestamp(
                    computer_sec,
                ).astimezone().replace(
                    microsecond=computer_nanosec // 1_000,
                ).isoformat(timespec='microseconds')
                writer.writerow((
                    frame_index,
                    (receipt_ns - first_receipt_ns) / 1e9,
                    acquisition_sec,
                    acquisition_nanosec,
                    acquisition_ns,
                    receipt_sec,
                    receipt_nanosec,
                    receipt_ns,
                    computer_ns,
                    computer_time,
                ))
        self.get_logger().info(
            f'Saved {len(rows)} camera timestamps to {timestamp_path}'
        )

    def save_maximum_power_image(self, image, value, output_stem):
        """Save the ROI frame nearest the maximum power sample."""
        image_path = output_stem.with_name(
            f'{output_stem.name}_max_power.png'
        )
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f'Could not write image {image_path}')
        self.get_logger().info(
            'Saved camera ROI nearest the maximum power '
            f'({value:g} W) to {image_path}'
        )

    @staticmethod
    def save_plot(rows, plot_path, minimum, maximum):
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        start_ns = rows[0][1]
        elapsed_times = [(row[1] - start_ns) / 1e9 for row in rows]
        values = [row[3] for row in rows]
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(elapsed_times, values, color='#21918c', linewidth=1.5)
        axis.axhline(
            minimum,
            color='#440154',
            linestyle='--',
            label=f'Min: {minimum:.6g} W',
        )
        axis.axhline(
            maximum,
            color='#fde725',
            linestyle='--',
            label=f'Max: {maximum:.6g} W',
        )
        axis.set_xlabel('Elapsed time [s]')
        axis.set_ylabel('Optical power [W]')
        axis.set_title('IMPT chip laser power over time')
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)

    def destroy_node(self):
        if self.recording:
            self.stop_recording()
        else:
            self.window_stop_event.set()
            if self.window_thread is not None:
                self.window_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PowerMeterSignalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
