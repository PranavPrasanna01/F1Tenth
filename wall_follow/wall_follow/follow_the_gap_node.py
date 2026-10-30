import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class FollowTheGapNode(Node):
    MAX_STEER = 0.4          # Steering clamp in radians (~±0.4 rad)
    GAP_THRESHOLD = 1.5      # Minimum distance (m) required to count as open gap
    MAX_RANGE_CLIP = 3.0     # Clip far distances to avoid 25m outdoor spikes

    def __init__(self):
        super().__init__('follow_the_gap_node')
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        
        self.max_speed = 1.0       # Matched to wall follower baseline
        self.min_speed = 0.5       # Speed in sharp turns
        self.bubble_radius = 0.4   # Safety bubble radius in meters
        self.last_logged_sec = 0


        self.get_logger().info('Follow the Gap Node initialized with logging & guards.')

    def preprocess_scan(self, ranges, angle_min, angle_increment):
        ranges = np.array(ranges, dtype = np.float32)
        
        # Replace NaN / Inf with 0.0
        ranges = np.nan_to_num(ranges, nan = 0.0, posinf = self.MAX_RANGE_CLIP, neginf = 0.0)
        
        # Crop to ±90 degrees field of view
        fov_min = -math.pi / 2
        fov_max = math.pi / 2
        
        start_idx = max(0, int((fov_min - angle_min) / angle_increment))
        end_idx = min(len(ranges) - 1, int((fov_max - angle_min) / angle_increment))
        
        cropped = ranges[start_idx:end_idx+1]
        
        # Clip max distances to 3.0m
        cropped = np.clip(cropped, 0.0, self.MAX_RANGE_CLIP)
        
        return cropped, start_idx

    def scan_callback(self, msg):
        # 1. Preprocess & Crop
        cropped_ranges, start_idx = self.preprocess_scan(msg.ranges, msg.angle_min, msg.angle_increment)
        if len(cropped_ranges) == 0:
            return

        # 2. Find closest valid obstacle (> 0.1m)
        valid_mask = cropped_ranges > 0.1
        if not np.any(valid_mask):
            return
        
        min_idx = np.argmin(np.where(valid_mask, cropped_ranges, np.inf))
        min_dist = cropped_ranges[min_idx]

        # 3. Apply Safety Bubble (Half-angle: asin(r / d))
        if min_dist > 0.05:
            half_angle = math.asin(min(1.0, self.bubble_radius / min_dist))
            beam_span = int(half_angle / msg.angle_increment)
            
            b_start = max(0, min_idx - beam_span)
            b_end = min(len(cropped_ranges) - 1, min_idx + beam_span)
            cropped_ranges[b_start:b_end+1] = 0.0

        # 4. Find the widest contiguous gap above GAP_THRESHOLD
        best_start, best_end = 0, -1
        current_start = None
        
        for i, dist in enumerate(cropped_ranges):
            if dist >= self.GAP_THRESHOLD:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    if (i - 1 - current_start) > (best_end - best_start):
                        best_start = current_start
                        best_end = i - 1
                    current_start = None
                    
        if current_start is not None and (len(cropped_ranges) - 1 - current_start) > (best_end - best_start):
            best_start = current_start
            best_end = len(cropped_ranges) - 1

        # 5. Target selection & Fallback guard
        max_val = np.max(cropped_ranges)
        if max_val == 0.0:
            # All beams zeroed out: straight ahead at min speed
            steering_angle = 0.0
            gap_width = 0
        elif best_end < best_start:
            # No gap above GAP_THRESHOLD: fallback to index with max range
            target_local_idx = np.argmax(cropped_ranges)
            gap_width = 1
            global_idx = target_local_idx + start_idx
            raw_angle = msg.angle_min + global_idx * msg.angle_increment
            steering_angle = max(-self.MAX_STEER, min(raw_angle, self.MAX_STEER))
        else:
            # Target center of widest gap
            target_local_idx = (best_start + best_end) // 2
            gap_width = (best_end - best_start) + 1
            global_idx = target_local_idx + start_idx
            raw_angle = msg.angle_min + global_idx * msg.angle_increment
            steering_angle = max(-self.MAX_STEER, min(raw_angle, self.MAX_STEER))

        # 6. Speed scaling against MAX_STEER (0.4 rad)
        steering_ratio = abs(steering_angle) / self.MAX_STEER
        speed = max(self.min_speed, self.max_speed * (1.0 - steering_ratio))

        # 7. Throttled Logging (Once per second)
        current_sec = msg.header.stamp.sec
        if current_sec != self.last_logged_sec:
            steer_deg = math.degrees(steering_angle)
            self.get_logger().info(
                f"min_dist: {min_dist:.2f}m | gap_beams: {gap_width} | target_angle: {steer_deg:.1f}° | speed: {speed:.2f}m/s"
            )
            self.last_logged_sec = current_sec

        # 8. Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'laser'
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)

        self.publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGapNode()
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
