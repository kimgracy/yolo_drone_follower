from setuptools import find_packages, setup

package_name = 'yolo_drone_follower'

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
    maintainer='kimgracy',
    maintainer_email='kimgracy@snu.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_drone_follower = yolo_drone_follower.yolo_drone_follower:main',
            'autonomous_yolo_follower = yolo_drone_follower.autonomous_yolo_follower:main'
        ],
    },
)
