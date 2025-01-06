from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, JointState
from moveit_msgs.msg import RobotState
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import subprocess

class RobotArm6DOF:
    def __init__(self):
        rospy.init_node('robot_arm_control', anonymous=True)

        # Load URDF into ROS Parameter Server
        urdf_path = "./robotarm.urdf"
        if not os.path.exists(urdf_path):
            rospy.logerr("URDF file not found at: %s", urdf_path)
            raise FileNotFoundError("URDF file not found.")

        with open(urdf_path, 'r') as urdf_file:
            urdf_content = urdf_file.read()
        rospy.set_param("robot_description", urdf_content)
        rospy.loginfo("URDF loaded into ROS Parameter Server.")

        # Launch RViz
        rviz_config_path = "./robotarm.rviz"
        if not os.path.exists(rviz_config_path):
            rospy.logwarn("RViz config file not found. Launching with default settings.")
            rviz_config_path = ""

        subprocess.Popen(["rosrun", "rviz", "rviz", "-d", rviz_config_path])

        # MoveIt! Initialization
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("manipulator")

        # Camera setup
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.camera_callback)

        # Joint State Publisher and Robot State
        self.joint_state_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.robot_state_pub = rospy.Publisher("/robot_states", RobotState, queue_size=10)

        # Calibration variables
        self.camera_to_robot_transform = None
        self.calibration_done = False

        # Initialize pose and detection result
        self.target_pose = Pose()
        self.detection_result = None

    def publish_joint_states(self, joint_values):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.group.get_active_joints()
        joint_state.position = joint_values
        self.joint_state_pub.publish(joint_state)

    def publish_robot_state(self):
        robot_state = RobotState()
        robot_state.joint_state.name = self.group.get_active_joints()
        robot_state.joint_state.position = self.group.get_current_joint_values()
        self.robot_state_pub.publish(robot_state)

    def camera_callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Object detection logic (use your preferred detection method)
            self.detection_result = self.detect_object(cv_image)

            if self.detection_result:
                rospy.loginfo("Object detected: %s", self.detection_result)

        except Exception as e:
            rospy.logerr("Error in camera callback: %s", str(e))

    def detect_object(self, image):
        # Placeholder detection logic
        detected = {"x": 0.5, "y": 0.2, "z": 0.1}
        return detected

    def calibrate_camera_to_robot(self, camera_points, robot_points):
        if len(camera_points) != len(robot_points):
            rospy.logerr("Calibration points mismatch: camera points and robot points must have the same number of samples.")
            return False

        try:
            camera_points_np = np.array(camera_points, dtype=np.float32)
            robot_points_np = np.array(robot_points, dtype=np.float32)
            _, rotation_vector, translation_vector = cv2.solvePnP(camera_points_np, robot_points_np, np.eye(3), None)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation_vector.squeeze()

            self.camera_to_robot_transform = transformation_matrix
            self.calibration_done = True

            rospy.loginfo("Calibration completed. Transformation Matrix:\n%s", self.camera_to_robot_transform)
            return True

        except Exception as e:
            rospy.logerr("Error during calibration: %s", str(e))
            return False

    def move_to_pose(self):
        if not self.detection_result:
            rospy.logwarn("No detection result available.")
            return

        if not self.calibration_done:
            rospy.logwarn("Calibration not completed. Cannot transform camera coordinates to robot coordinates.")
            return

        camera_point = np.array([self.detection_result['x'], self.detection_result['y'], self.detection_result['z'], 1])
        robot_point = np.dot(self.camera_to_robot_transform, camera_point)[:3]

        # Set target pose based on transformed coordinates
        self.target_pose.position.x = robot_point[0]
        self.target_pose.position.y = robot_point[1]
        self.target_pose.position.z = robot_point[2]

        # Orientation can be predefined or computed
        self.target_pose.orientation.w = 1.0

        self.group.set_pose_target(self.target_pose)

        # Plan and execute
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        # Publish joint and robot states
        self.publish_joint_states(self.group.get_current_joint_values())
        self.publish_robot_state()

        if plan:
            rospy.loginfo("Move successful!")
        else:
            rospy.logwarn("Move failed.")

if __name__ == "__main__":
    try:
        robot_arm = RobotArm6DOF()

        # Example calibration points
        camera_points = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]
        robot_points = [[0.5, 0.2, 0.3], [0.6, 0.2, 0.3], [0.5, 0.3, 0.3]]

        robot_arm.calibrate_camera_to_robot(camera_points, robot_points)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
