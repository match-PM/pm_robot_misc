# pm_robot_misc
Collection of different components working with the pm ros2 framework.

## IMPT chip laser signal nodes

Build and launch the camera evaluator, PM100USB driver, and power evaluator:

```bash
colcon build --packages-select \
  pm_impt_chip_laser_signal \
  pm_impt_chip_laser_signal_powermeter \
  thorlabs_pm100usb_driver
source install/setup.bash
ros2 launch pm_impt_chip_laser_signal_powermeter \
  impt_chip_laser_signals.launch.py
```

The power evaluator subscribes to `thorlabs_pm100usb/power` and republishes
timestamped `pm_skills_interfaces/msg/Float64Stamped` messages on
`pm_impt_chip_laser_signal_powermeter/float_alignment`. It also passively
subscribes to `Camera_Top_View/pylon_ros2_camera_node/image_raw`. When the
stamped power output has a subscriber, it displays a live one-dimensional
power scale and records every received power sample and unprocessed camera
frame after applying the same centered 30-by-30-percent ROI as the camera
signal evaluator. Disconnecting the last subscriber (or stopping the node)
saves the power CSV and PNG plot, an MP4 ROI camera stream, a camera timestamp
CSV, and the ROI camera frame nearest the maximum power sample as a separate
PNG. All files from a session share the same filename timestamp.

The `input_topic`, `camera_topic`, `output_topic`, `output_directory`, and
`camera_video_fps` node parameters can be overridden when needed.
The ROI parameters are also exposed as `roi_center_x`, `roi_center_y`,
`roi_height`, and `roi_width`; their defaults are `0`, `0`, `30`, and `30` to
match `pm_impt_chip_laser_signal`.
