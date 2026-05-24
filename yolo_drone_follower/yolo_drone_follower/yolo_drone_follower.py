import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# PX4 공식 토픽 메시지 구조들
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus

class YoloDroneFollower(Node):
    def __init__(self):
        super().__init__('yolo_drone_follower')

        # 1. YOLO 및 추적 모델 인스턴스 초기화
        self.model_path = '/home/kimgracy/KIST/ros2_ws/src/yolo_drone_follower/yolo_drone_follower/dead_bird.pt'
        self.model = YOLO(self.model_path)
        self.target_class_name = 'dead-bird'

        # 제어 게인 설정
        self.target_width = 150.0   # 타겟 사물의 목표 가로 픽셀 크기
        self.kp_x = 0.005           # 전진 속도 게인 (m/s)
        self.kp_z = 0.003           # 상하 속도 게인 (m/s)
        self.kp_yaw = 0.004         # 회전(Yaw) 속도 게인 (rad/s)

        self.bridge = CvBridge()
        
        # 기체 상태 변수 선언 (타겟 미발견 시 0.0 호버링 상태가 기본값)
        self.nav_state = None
        self.target_detected = False
        self.cmd_vx = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0

        # PX4 VehicleStatus 바인딩을 위한 고유 QoS 세팅
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. 퍼블리셔 선언
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
        # 서브스크라이버 선언
        self.vehicle_status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # 3. 메인 제어 타이머 루프 (20Hz = 0.05초)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('YOLO 드론 제어 노드가 가동되었습니다. 카메라 스트리밍을 대기합니다.')

    def vehicle_status_callback(self, msg):
        # 현재 PX4 내부 비행 모드 동기화 (Offboard 상태인지 모니터링 목적)
        self.nav_state = msg.nav_state

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 에러: {e}')
            return

        img_height, img_width, _ = cv_image.shape
        img_center_x = img_width / 2.0
        img_center_y = img_height / 2.0

        # YOLOv8 추론 연산 실행 (ValueError 이슈 수정으로 'cpu' 지정, 필요시 하드웨어 점검 후 'cuda' 가동)
        results = self.model(cv_image, verbose=False, device='cpu')
        
        self.target_detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                
                if label == self.target_class_name:
                    xyxy = box.xyxy[0].to('cpu').numpy()
                    xmin, ymin, xmax, ymax = xyxy
                    
                    bbox_center_x = (xmin + xmax) / 2.0
                    bbox_center_y = (ymin + ymax) / 2.0
                    bbox_width = xmax - xmin
                    
                    self.target_detected = True
                    
                    # 조향 오차에 기반한 속도 계산 가동
                    self.cmd_yaw_rate = float((img_center_x - bbox_center_x) * self.kp_yaw)
                    
                    error_width = self.target_width - bbox_width
                    self.cmd_vx = float(error_width * self.kp_x)
                    if self.cmd_vx < 0.0: self.cmd_vx = 0.0  # 안전을 위한 후진 봉쇄
                    
                    # 이미지 상단이 -Y이므로 NED 상 고도 상승(-Z) 매핑을 위한 가감산 조율
                    self.cmd_vz = float(-(img_center_y - bbox_center_y) * self.kp_z)

                    # 물리 보호용 하드웨어 최대 제한 가중치 (Saturation)
                    self.cmd_vx = min(self.cmd_vx, 0.5)
                    self.cmd_vz = np.clip(self.cmd_vz, -0.3, 0.3)
                    self.cmd_yaw_rate = np.clip(self.cmd_yaw_rate, -0.3, 0.3)

                    # 화면 디스플레이 가이드박스 페인팅
                    cv2.rectangle(cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"{label} Tracking", (int(xmin), int(ymin)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
            if self.target_detected: break

        if not self.target_detected:
            # 화면 내 타겟 실종 시 관성 드리프트를 막기 위해 완전 제자리 정지 유도
            self.cmd_vx = 0.0
            self.cmd_vz = 0.0
            self.cmd_yaw_rate = 0.0

        # 모니터에 영상 출력 윈도우 생성
        cv2.imshow("YOLO Drone Follower", cv_image)
        cv2.waitKey(1)

    def timer_callback(self):
        """
        [원인 해결의 핵심부]
        PX4 하트비트와 속도 셋포인트를 기체의 실제 비행 모드 상태에 상관없이
        20Hz 주기로 '무조건 동시에' 끊임없이 밀어 넣습니다.
        이 구조가 유지되어야 QGC에서 비행 모드를 전환하려 할 때 셋포인트를 신뢰하고 오프보드가 잠금 해제됩니다.
        """
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True   # 속도 제어 인터페이스 개방
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        """
        PX4 가이드라인을 정확히 준수하는 오프보드 속도 셋포인트 함수.
        사용하지 않는 position, acceleration 배열에는 엄격하게 NaN(Not a Number) 처리를 해야
        PX4의 비행 안정 제약 시스템이 입력을 오작동으로 판단해 튕겨내는 문제를 해결할 수 있습니다.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # 위치 및 가속도 제어 미사용 명시 처리
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        
        # 드론 로컬 기준(바디 프레임 대응용 매핑) 전진 및 상하 속도 설정
        msg.velocity[0] = float(self.cmd_vx)       # X: 전진 속도
        msg.velocity[1] = 0.0                      # Y: 측면 횡이동은 배제
        msg.velocity[2] = float(self.cmd_vz)       # Z: 승하강 속도 (NED 좌표계: 위 방향은 -값)
        
        # 각도 제어 미사용 지정 및 회전각 속도(Yaw rate) 전달
        msg.yaw = float('nan')                     
        msg.yawspeed = float(self.cmd_yaw_rate)    # Z축 기준 기체 회전 속도 (rad/s)
        
        self.trajectory_setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDroneFollower()
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