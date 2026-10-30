import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.signals import SignalHandlerOptions

class WallFollowNode(Node):
    LOOKAHEAD_L = 1.0          # Lookahead distance (L from lecture)
    MAX_STEERING_ANGLE = 0.4   # Steering clamp in radians (~±0.4 rad)

    def __init__(self):
        super().__init__('wall_follow_node')
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        
        # Controller parameters (Dropped Kp to 0.4 for safe initial tuning)
        self.desired_distance = 1.0  # meters
        self.kp = 0.4
        self.kd = 0.0                # Keep at 0 until oscillation starts
        
        self.prev_error = 0.0
        self.prev_time = None
        self.last_logged_sec = 0     # Throttle tracking for 1-second logs

        self.get_logger().info('Wall Follow Node initialized with comprehensive logging.')

    def get_beam_index(self, angle, scan_msg):
        if scan_msg.angle_increment == 0:
            return 0
        index = int(round((angle - scan_msg.angle_min) / scan_msg.angle_increment))
        return max(0, min(index, len(scan_msg.ranges) - 1))

    def scan_callback(self, msg):
        # b = perpendicular beam (90°), a = beam angled ahead (theta = 50°)
        theta_angle = math.radians(50.0)
        b_angle = math.radians(90.0)
        
        idx_a = self.get_beam_index(theta_angle, msg)
        idx_b = self.get_beam_index(b_angle, msg)
        
        a = msg.ranges[idx_a]
        b = msg.ranges[idx_b]
        
        # Note: Sim uses ~25m for max range/no hit instead of inf
        if math.isnan(a) or math.isnan(b):
            return

        # Lecture 4 Geometry calculations
        theta_diff = b_angle - theta_angle
        numerator = a * math.cos(theta_diff) - b
        denominator = a * math.sin(theta_diff)
        
        alpha = 0.0
        if denominator != 0:
            alpha = math.atan(numerator / denominator)
            current_distance = b * math.cos(alpha)
            projected_distance = current_distance + self.LOOKAHEAD_L * math.sin(alpha)
        else:
            projected_distance = a

        alpha_deg = math.degrees(alpha)

        # Error and Steering
        error = self.desired_distance - projected_distance

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_time is None or self.kd == 0.0:
            dt = 0.05
            derivative = 0.0
        else:
            dt = current_time - self.prev_time
            if dt <= 0.0:
                dt = 0.001
            derivative = (error - self.prev_error) / dt
        
        self.prev_time = current_time
        self.prev_error = error

        raw_steering = (self.kp * error) + (self.kd * derivative)
        steering_angle = -raw_steering  # Corrected sign for left wall

        # Clamp Steering Angle
        steering_angle = max(-self.MAX_STEERING_ANGLE, min(steering_angle, self.MAX_STEERING_ANGLE))

        # ==========================================
        # Throttled Logging (Once per second)
        # ==========================================
        current_sec = msg.header.stamp.sec
        if current_sec != self.last_logged_sec:
            self.get_logger().info(
                f"a: {a:.2f}m | b: {b:.2f}m | alpha: {alpha_deg:.1f}° | "
                f"proj: {projected_distance:.2f}m | err: {error:.2f} | steer: {steering_angle:.2f}"
            )
            self.last_logged_sec = current_sec

        # Publish Drive Command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'laser'
        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = steering_angle

        self.publisher.publish(drive_msg)

def main(args = None):
    rclpy.init(args = args, signal_handler_options=SignalHandlerOptions.NO)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        for _ in range(3):
            node.publisher.publish(stop)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
