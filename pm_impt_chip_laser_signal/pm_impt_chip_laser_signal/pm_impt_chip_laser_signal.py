import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64, 'topic', 10)
        timer_period = 0.5  # seconds (publish every 0.5s for a smooth sine wave)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = time.monotonic()
        self.frequency = 0.1  # Hz

    def timer_callback(self):
        msg = Float64()
        elapsed = time.monotonic() - self.start_time
        msg.data = math.sin(2.0 * math.pi * self.frequency * elapsed)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%f"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()