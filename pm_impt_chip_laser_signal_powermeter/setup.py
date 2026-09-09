import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'pm_impt_chip_laser_signal_powermeter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mll',
    maintainer_email='terei@match.uni-hannover.de',
    description=(
        'Timestamp, visualize, and record the IMPT chip laser power signal.'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'pm_impt_chip_laser_signal_powermeter = '
            'pm_impt_chip_laser_signal_powermeter.'
            'pm_impt_chip_laser_signal_powermeter:main',
        ],
    },
)
