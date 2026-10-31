import math

import numpy as np
import rclpy
import torch

from geometry_msgs.msg import Twist
from rclpy.node import Node
from nav_msgs.msg import Odometry
from lerobot.common.policies.act.modeling_act import ACTPolicy


MODEL_PATH = "/home/ubuntu/checkpoints/train/diff_drive_act"


def quaternion_to_yaw(x, y, z, w):
    """QuaternionからYaw角を計算する。"""
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class DiffDriveACTNode(Node):
    def __init__(self):
        super().__init__("diff_drive_act_inference")

        self.device = torch.device("cpu")

        # ACTモデルをロード
        self.get_logger().info("Loading ACT model...")
        self.policy = ACTPolicy.from_pretrained(MODEL_PATH)
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info("ACT model loaded.")

        # Odometryを購読
        self.subscription = self.create_subscription(
            Odometry,
            "/model/demoRobot/odometry",
            self.odom_callback,
            10,
        )

        self.latest_state = None

        # 1秒ごとに推論結果を表示
        self.timer = self.create_timer(1.0, self.inference_callback)

        self.cmd_pub = self.create_publisher(
                Twist, "/cmd_vel", 10
        )

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist

        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        # 学習時と同じ6次元の状態
        self.latest_state = np.array(
            [
                position.x,
                position.y,
                math.sin(yaw),
                math.cos(yaw),
                velocity.linear.x,
                velocity.angular.z,
            ],
            dtype=np.float32,
        )

    def inference_callback(self):
        if self.latest_state is None:
            self.get_logger().warning("Waiting for Odometry...")
            return

        state = torch.from_numpy(self.latest_state).unsqueeze(0)

        obs = {
            "observation.state": state,
            "observation.environment_state": state.clone(),
        }

        # ACT推論
        with torch.inference_mode():
            action = self.policy.select_action(obs)

        action = action.squeeze(0).cpu().numpy()

        if not np.isfinite(action).all():
            self.get_logger().error("Invalid action detected.")
            return

        linear_velocity = float(action[0])
        angular_velocity = float(action[1])

        self.get_logger().info(
            f"State: {self.latest_state.tolist()} | "
            f"Predicted action: "
            f"linear={linear_velocity:.4f}, "
            f"angular={angular_velocity:.4f}"
        )

        # ACTの予測値をGazeboに送信
        cmd = Twist()
        cmd.linear.x = float(np.clip(linear_velocity, -0.2, 0.2))
        cmd.angular.z = float(np.clip(angular_velocity, -0.3, 0.3))

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Command: linear={cmd.linear.x:.3f}, "
            f"angular={cmd.angular.z:.3f}"
        )


def main():
    rclpy.init()

    node = DiffDriveACTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
