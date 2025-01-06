import numpy as np
import tf
import rospy
from tf.transformations import translation_matrix, quaternion_matrix

class Calibration:
    def __init__(self):
        rospy.init_node("camera_robot_calibration")

        # Transformation: Camera -> End-Effector
        self.camera_to_effector = None

        # Listener for TF
        self.tf_listener = tf.TransformListener()

    def get_transformation(self, from_frame, to_frame):
        try:
            self.tf_listener.waitForTransform(from_frame, to_frame, rospy.Time(), rospy.Duration(4.0))
            trans, rot = self.tf_listener.lookupTransform(from_frame, to_frame, rospy.Time())
            trans_mat = translation_matrix(trans)
            rot_mat = quaternion_matrix(rot)
            return np.dot(trans_mat, rot_mat)
        except Exception as e:
            rospy.logerr(f"Error getting transform from {from_frame} to {to_frame}: {e}")
            return None

    def calibrate(self):
        # Example: Get Camera to End-Effector Transformation
        self.camera_to_effector = self.get_transformation("camera_link", "end_effector")
        if self.camera_to_effector is not None:
            rospy.loginfo(f"Calibration Matrix:\n{self.camera_to_effector}")
        else:
            rospy.logwarn("Calibration failed.")

if __name__ == "__main__":
    calibration = Calibration()
    calibration.calibrate()