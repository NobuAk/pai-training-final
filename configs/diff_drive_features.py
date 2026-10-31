features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "sin_yaw", "cos_yaw", "linear_velocity", "angular_velocity"],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "sin_yaw", "cos_yaw", "linear_velocity", "angular_velocity"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["linear_velocity", "angular_velocity"],
    },
}

robot_type = "diff_drive"
repo_id = "diff_drive_dataset"
task_name = "drive_left_turn"
