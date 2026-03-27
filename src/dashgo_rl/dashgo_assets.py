# dashgo_assets.py
# 2026-01-24: Fixed Config Collision (joint_drive vs actuators)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from .project_paths import DASHGO_URDF_PATH

# 重构后统一从项目配置目录加载 URDF，避免继续引用旧的源码同级路径。
URDF_PATH = str(DASHGO_URDF_PATH)

DASHGO_D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        make_instanceable=False,
        activate_contact_sensors=True,

        # [关键参数 1] 刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
        ),

        # [关键参数 2] 关节属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),

        # [⚠️ 核心修复] 必须显式设为 None！
        # [架构师修复 2026-01-24] 解决新旧配置冲突问题
        # 问题：如果使用 actuators，必须显式禁用 spawn 内部的 joint_drive
        # 原因：同时应用"旧驱动"和"新电机"会导致参数验证失败
        # 修复：设置 joint_drive=None，彻底禁用默认驱动
        joint_drive=None,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # 抬高到15cm，防止轮子陷入地下（轮子半径6.32cm）
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),

    # [核心动力源] 你的强力电机配置
    # [架构师修正 2026-01-24] 修复机器人"瘫痪"问题
    # 问题：damping=5.0 太低，机器人响应太慢
    # 解决：damping=100.0，增强速度跟踪能力
    actuators={
        "diff_drive": ImplicitActuatorCfg(
            # [验证] URDF 关节名称：left_wheel_joint, right_wheel_joint
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],

            # 强力参数（保持之前的修复）
            effort_limit=20.0,     # 最大扭矩 20 N·m
            velocity_limit=10.0,    # 最大转速 10 rad/s ≈ 0.64 m/s
            stiffness=0.0,          # 速度控制时刚度设为0
            damping=100.0,          # 从 5.0 提高到 100.0（增强速度跟踪）
        ),
    },
)
