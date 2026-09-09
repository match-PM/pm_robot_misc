"""Launch both IMPT signal nodes and the Thorlabs PM100USB driver."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Create the three-node IMPT laser signal launch description."""
    driver_share = get_package_share_directory('thorlabs_pm100usb_driver')
    driver_parameters = os.path.join(
        driver_share,
        'config',
        'pm100usb.yaml',
    )
    return LaunchDescription([
        Node(
            package='thorlabs_pm100usb_driver',
            executable='thorlabs_pm100usb_driver',
            name='thorlabs_pm100usb',
            output='screen',
            parameters=[driver_parameters],
        ),
        # Node(
        #     package='pm_impt_chip_laser_signal',
        #     executable='pm_impt_chip_laser_signal',
        #     output='screen',
        # ),
        Node(
            package='pm_impt_chip_laser_signal_powermeter',
            executable='pm_impt_chip_laser_signal_powermeter',
            output='screen',
        ),
    ])
