from setuptools import find_packages, setup

package_name = 'pm_impt_chip_laser_signal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mll',
    maintainer_email='terei@match.uni-hannover.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pm_impt_chip_laser_signal = pm_impt_chip_laser_signal.pm_impt_chip_laser_signal:main'
        ],
    },
)
