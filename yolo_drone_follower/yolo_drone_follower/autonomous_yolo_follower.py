#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
from datetime import datetime

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition

class AutonomousYoloFollower(Node):
    def __init__(self):
        super().__init__('autonomous_yolo_follower')

        # 1. YOLO Model Configuration
        self.model_path = '/home/kimgracy/KIST/ros2_ws/src/yolo_drone_follower/yolo_drone_follower/dead_bird.pt'
        self.model = YOLO(self.model_path)
        self.target_class_name = 'dead-bird'

        # Control Parameters and Thresholds
        self.target_width = 150.0   
        self.kp_x = 0.005           
        self.kp_z = 0.003           
        self.kp_yaw = 0.004         
        
        self.yolo_hz = 3            # Base Hz used for timeout calculation
        self.quick_time = 0.5       # Required detection duration to trigger offboard (seconds)
        self.focus_time = 1.5       # Validation time after alignment (seconds)
        self.mc_acceptance_radius = 0.3  # Position acceptance radius (meters)
        self.heading_acceptance_angle = 0.05  # Yaw heading error tolerance (radians)
        self.align_emergency_threshold = 60   # Alignment timeout (tick count)

        self.bridge = CvBridge()
        
        # Vehicle Status
        self.nav_state = None
        self.arming_state = None    
        self.current_yaw = 0.0      
        self.current_z = 0.0        
        self.pos = np.array([0.0, 0.0, 0.0]) # NED Local Position [X, Y, Z]
        self.vel = np.array([0.0, 0.0, 0.0]) # NED Local Velocity [Vx, Vy, Vz]
        
        self.target_detected = False
        self.obstacle_label = ''
        self.obstacle_x = 0.0
        self.image_size = (640, 480)
        self.current_ratio_percent = 0.0 # Variable to store real-time target ratio within the image frame
        
        self.sys_id = 1
        self.comp_id = 1
        
        # Phase Setup
        self.phase = 0
        self.subphase = 'before flight'
        
        self.bird_count = 0
        self.time_count = 0
        self.yolo_time_count = 0
        self.emergency_time_checker = 0

        self.cmd_vx = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0
        self.goal_position = np.array([0.0, 0.0, 0.0])
        self.goal_yaw = 0.0

        # Approach Control Parameters
        self.approach_speed = 0.5  # Forward approach speed (m/s)

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Logging Setup
        log_dir = os.path.join(os.getcwd(), 'flight_logs')
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = os.path.join(log_dir,  f'log_{current_time}.txt')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # Publishers
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        # Subscribers
        self.vehicle_status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.local_pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_position_callback, qos_profile)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10) 

        self.timer = self.create_timer(0.05, self.main_timer_callback)
        self.print('Autonomous YOLO Follower Node Initiated (Mission-to-Offboard Intercept Trigger Active)')


    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        self.logger.info(*args, **kwargs)


    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state


    def local_position_callback(self, msg):
        self.current_yaw = msg.heading
        self.current_z = msg.z  
        self.pos = np.array([msg.x, msg.y, msg.z])
        self.vel = np.array([msg.vx, msg.vy, msg.vz])


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        img_height, img_width, _ = cv_image.shape
        self.image_size = (img_width, img_height)
        img_center_x = img_width / 2.0
        img_center_y = img_height / 2.0

        results = self.model(cv_image, verbose=False, device='cpu')
        current_target_found = False
        ratio_percent = 0.0

        for result in results:
            for box in result.boxes:
                if self.model.names[int(box.cls[0])] == self.target_class_name:
                    xyxy = box.xyxy[0].to('cpu').cpu().numpy()
                    xmin, ymin, xmax, ymax = xyxy
                    bbox_center_x = (xmin + xmax) / 2.0
                    bbox_center_y = (ymin + ymax) / 2.0
                    bbox_width = xmax - xmin
                    current_target_found = True
                    
                    self.obstacle_label = self.target_class_name
                    self.obstacle_x = bbox_center_x
                    
                    ratio_percent = (bbox_width / img_width) * 100.0
                    
                    # Bounding Box Visualization
                    cv2.rectangle(cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    confidence = float(box.conf[0])
                    label_text = f"{self.target_class_name} {confidence:.2f} ({ratio_percent:.1f}%)"
                    cv2.putText(cv_image, label_text, (int(xmin), max(int(ymin) - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Update internal tracking variable for control loops
                    self.current_ratio_percent = ratio_percent
                    break
            if current_target_found: 
                break

        self.target_detected = current_target_found
        if not self.target_detected:
            self.current_ratio_percent = 0.0
        
        # Display Texts
        status_text = f"PHASE {self.phase} - {self.subphase.upper()}"
        cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        display_alt = -self.current_z
        alt_text = f"Current Altitude: {display_alt:.2f}m"
        cv2.putText(cv_image, alt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.target_detected:
            cv2.putText(cv_image, "TARGET LOCK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ratio_text = f"BB Width Ratio: {ratio_percent:.1f}%"
            cv2.putText(cv_image, ratio_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(cv_image, "BB Width Ratio: 0.0%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        cv2.imshow("YOLO Autonomous Tracker", cv_image)
        cv2.waitKey(1)


    def main_timer_callback(self):
        if self.nav_state is None:
            return

        # Background Offboard Heartbeat
        self.publish_offboard_control_mode()

        # -----------------------------------------------------------------
        # PHASE 0 : QGC Takeoff
        # -----------------------------------------------------------------
        if self.phase == 0:
            if self.nav_state in [3, 5, 17]:
                self.print(f"Takeoff Complete... PX4 nav_state: {self.nav_state}")
                self.phase = 1
                self.subphase = 'survey mission'
                self.print('\n[Phase 0 -> 1] Takeoff detected. Moving to PX4 Mission Modes.\n')

        # -----------------------------------------------------------------
        # PHASE 1 : YOLO Detection during Mission Mode
        # -----------------------------------------------------------------
        elif self.phase == 1:
            if self.obstacle_label == 'dead-bird':
                self.obstacle_label = ''
                self.bird_count += 1
                
                if self.bird_count >= self.yolo_hz * self.focus_time:
                    self.print('Target Detected\n')
                    self.bird_count = 0
                    self.goal_position = self.get_braking_position(self.pos, self.vel)
                    
                    self.request_offboard_mode()
                    
                    self.phase = 2
                    self.subphase = 'pause'
                    self.print('\n[Phase 1 -> 2] Transitioning to Offboard Flight Mode.\n')
            else:
                self.bird_count = max(0, self.bird_count - 1)

        # -----------------------------------------------------------------
        # PHASE 2 : YOLO following after Offboard Transition
        # -----------------------------------------------------------------
        elif self.phase == 2:
            
            if self.subphase == 'pause':
                self.publish_trajectory_setpoint(position_sp=self.goal_position)
                if np.linalg.norm(self.pos - self.goal_position) < self.mc_acceptance_radius:
                    self.goal_yaw = self.get_bearing_to_target()
                    self.emergency_time_checker = 0
                    self.subphase = 'align'
                    self.print('[Subphase : pause -> align] Aligning drone heading to target.')

            elif self.subphase == 'align':
                self.publish_trajectory_setpoint(position_sp=self.goal_position, yaw_sp=self.goal_yaw)
                
                if np.abs((self.current_yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi) < self.heading_acceptance_angle:
                    self.bird_count = 0
                    self.time_count = 0
                    self.yolo_time_count = 0
                    self.subphase = 'detecting obstacle'
                    self.print('[Subphase : align -> detecting obstacle] Starting target validation.')
                else:
                    self.emergency_time_checker += 1
                    if self.emergency_time_checker >= self.align_emergency_threshold:
                        self.emergency_time_checker = 0
                        self.subphase = 'pause' 
                        self.print('[Warning] Alignment timeout. Re-stabilizing.')

            elif self.subphase == 'detecting obstacle':
                self.publish_trajectory_setpoint(position_sp=self.goal_position, yaw_sp=self.goal_yaw)
                
                if self.time_count >= self.yolo_hz * self.focus_time:
                    bird_detect_ratio = self.bird_count / max(1, self.yolo_time_count)
                    
                    if bird_detect_ratio >= 0.30:
                        self.bird_count = 0
                        self.time_count = 0
                        self.yolo_time_count = 0
                        
                        # Once target is confirmed, transition to forward approach phase instead of landing directly
                        self.subphase = 'approaching target'
                        self.print('[Subphase : Validated] Target confirmed! Approaching target while aligning yaw.')
                    else:
                        self.bird_count = 0
                        self.time_count = 0
                        self.yolo_time_count = 0
                        self.phase = 1
                        self.subphase = 'monitoring mission'
                        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0)
                        self.print('[Validated Failed] False alarm. Handing over control back to PX4 AUTO_MISSION.')
                else:
                    self.time_count += 1
                    if self.obstacle_label == 'dead-bird':
                        self.bird_count += 1
                        self.obstacle_label = ''
                    self.yolo_time_count += 1

            elif self.subphase == 'approaching target':
                # Check if the bounding box width ratio has reached or exceeded 50%
                if self.target_detected and self.current_ratio_percent >= 50.0:
                    # Halt forward progression, trigger landing command, and shift state
                    self.publish_velocity_setpoint(0.0, 0.0, 0.0, yaw_rate=0.0)
                    self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.subphase = 'landing'
                    self.print('\n[Approaching -> Landing] Target is close enough (>= 50%). Triggering AUTO LAND mode.\n')
                else:
                    # Real-time computation of target bearing angle
                    if self.target_detected:
                        self.goal_yaw = self.get_bearing_to_target()
                    
                    # Compute Body-Fixed forward velocity relative to current drone heading (current_yaw)
                    # Coordinate transform needed as PX4 Offboard Velocity Setpoint operates in NED frame
                    v_north = self.approach_speed * np.cos(self.current_yaw)
                    v_east = self.approach_speed * np.sin(self.current_yaw)
                    v_down = 0.0  # Maintain altitude during horizontal transit
                    
                    # Publish velocity setpoint incorporating target heading tracking
                    self.publish_velocity_setpoint(v_north, v_east, v_down, yaw=self.goal_yaw)
                    self.print(f"[Approaching] Width: {self.current_ratio_percent:.1f}% / 50.0% | Aligning Yaw to {self.goal_yaw:.2f} rad", end='\r')

            elif self.subphase == 'landing':
                self.publish_velocity_setpoint(0.0, 0.0, 0.0, yaw_rate=0.0)
                self.print('[Landing] PX4 Landing sequence initiated. Monitoring altitude...', end='\r')

    # -----------------------------------------------------------------
    # Geometric Operation Helper Sub-methods for Precision Maneuvers
    # -----------------------------------------------------------------
    def get_braking_position(self, current_pos, current_vel):
        braking_distance_factor = 0.8 
        return current_pos + current_vel * braking_distance_factor

    def get_bearing_to_target(self):
        center_error = self.obstacle_x - (self.image_size[0] / 2.0)
        yaw_error_rad = -(center_error / (self.image_size[0] / 2.0)) * 0.4 
        return self.normalize_angle(self.current_yaw + yaw_error_rad)

    def normalize_angle(self, angle):
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

    # -----------------------------------------------------------------
    # PX4 Communication and Transmission Low-level Interface Meta
    # -----------------------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True  
        msg.velocity = True   
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_velocity_setpoint(self, vx, vy, vz, yaw=float('nan'), yaw_rate=float('nan')):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float('nan'), float('nan'), float('nan')] 
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = float(yaw)
        msg.yawspeed = float(yaw_rate)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_trajectory_setpoint(self, position_sp, yaw_sp=float('nan')):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(position_sp[0]), float(position_sp[1]), float(position_sp[2])]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw_sp)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_zero_setpoint(self):
        self.publish_velocity_setpoint(0.0, 0.0, 0.0, yaw=float('nan'), yaw_rate=float('nan'))

    def request_offboard_mode(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def send_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.target_system = self.sys_id
        msg.target_component = self.comp_id
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousYoloFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()