# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
#from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"D:\\code\\isaaclab_ws\\PM01_Walk\\source\\PM01_Walk\\PM01_Walk\\assets\\robots\\pm01\\usd\\pm01.usd",
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     rigid_body_enabled=True,
        #     enable_gyroscopic_forces=True,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=100.0,
        # ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True,
        #     solver_position_iteration_count=4,
        #     solver_velocity_iteration_count=1,
        #     sleep_threshold=0.005,
        #     stabilization_threshold=0.001,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), 
        joint_pos={
            "j00_hip_pitch_l": -0.3,   # 左髋前屈
            "j03_knee_pitch_l": 0.6,   # 左膝弯曲
            "j04_ankle_pitch_l": -0.3, # 左踝背屈（脚尖稍下压）

            "j06_hip_pitch_r": -0.3,   # 右髋前屈
            "j09_knee_pitch_r": 0.6,   # 右膝弯曲
            "j10_ankle_pitch_r": -0.3, # 右踝背屈
        }
    ),
    soft_joint_pos_limit_factor=0.9,
    # actuators={
    #     # "cart_actuator": ImplicitActuatorCfg(
    #     #     joint_names_expr=["slider_to_cart"],
    #     #     effort_limit_sim=400.0,
    #     #     stiffness=0.0,
    #     #     damping=10.0,
    #     # ),
    #     # "pole_actuator": ImplicitActuatorCfg(
    #     #     joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
    #     # ),
    # },
    actuators={
    "default": ImplicitActuatorCfg(
        joint_names_expr=[".*"],     # ✅ 匹配全部关节
        effort_limit_sim=200.0,      # 力矩上限，可稍大点以防漂移
        stiffness=50.0,             # 高刚度 -> 僵硬
        damping=5.0,                 # 阻尼 -> 稳定
    ),
},
)
"""Configuration for a simple Cartpole robot."""