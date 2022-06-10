from setuptools import setup
import os
from glob import glob

package_name = 'car_detector_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'models', 'utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='bjh009@ucsd.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_detector = car_detector_pkg.car_detector:main'
        ],
    },
)
