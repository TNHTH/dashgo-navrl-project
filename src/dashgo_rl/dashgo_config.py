"""
DashGo机器人配置中心

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
参数来源: ROS配置文件 (drivers/EAI_DRIVER/src/config/)

用途:
    统一管理DashGo机器人的所有物理参数，确保仿真与实物对齐。
    所有参数从ROS配置文件读取，避免硬编码。

更新时间: 2026-01-26
机器人型号: DashGo D1
制造商: EAI (eaibot.cn)
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional

from .project_paths import EAI_PARAMS_YAML


def _unwrap_ros_parameter_payload(payload: dict | None) -> dict:
    """兼容 ROS2 driver YAML 与旧扁平 YAML。"""
    if not isinstance(payload, dict):
        return {}
    if "dashgo_driver_node" in payload:
        node_payload = payload.get("dashgo_driver_node") or {}
        ros_params = node_payload.get("ros__parameters")
        if isinstance(ros_params, dict):
            return ros_params
    return payload


# =============================================================================================
# 常量配置字典（用于奖励函数等）
# =============================================================================================

MOTION_CONFIG = {
    "max_lin_vel": 0.3,          # 最大线速度 (m/s)
    "min_lin_vel": -0.3,         # 最小线速度 (m/s)
    "max_ang_vel": 1.0,          # 最大角速度 (rad/s)
    "min_ang_vel": -1.0,         # 最小角速度 (rad/s)
    "max_lin_acc": 1.0,          # 最大线加速度 (m/s²)
    "max_ang_acc": 0.6,          # 最大角加速度 (rad/s²)
    "min_in_place_vel": 0.4,     # 原地旋转速度 (rad/s)
}

REWARD_CONFIG = {
    "high_speed_threshold": 0.25,    # 高速阈值 (m/s)
    "low_speed_threshold": 0.05,     # 低速阈值 (m/s)
}

LIDAR_CONFIG = {
    "max_range": 12.0,            # 最大检测距离 (m) - 仿真配置
    "min_range": 0.15,            # 最小检测距离 (m)
    "update_rate": 10.0,          # 更新频率 (Hz)
    "num_channels": 216,          # 仿真射线数（前向180°双相机拼接）
    "horizontal_fov": 180.0,      # 水平视野 (度)
    "install_height": 0.13,       # 安装高度 (m)
}

NAVIGATION_CONFIG = {
    "xy_goal_tolerance": 0.2,     # 位置公差 (m)
    "yaw_goal_tolerance": 0.1,    # 姿态公差 (rad)
    "obstacle_range": 3.0,        # 障碍物检测距离 (m)
    "raytrace_range": 3.5,        # 障碍物清除距离 (m)
}


# =============================================================================================
# 机器人物理尺寸参数
# =============================================================================================

@dataclass
class DashGoPhysicalSpecs:
    """
    DashGo D1 物理尺寸规格

    来源: URDF模型 + 官方规格文档
    """
    # 机器人外形
    body_diameter: float = 0.406      # 主体直径 (m)
    body_height: float = 0.21          # 主体高度 (m)
    body_mass: float = 13.7            # 主体质量 (kg)
    robot_radius: float = 0.2          # 机器人半径 (m) - 用于导航

    # 轮子参数
    wheel_diameter: float = 0.1264     # 轮子直径 (m) - ROS配置
    wheel_width: float = 0.04          # 轮子宽度 (m)
    wheel_mass: float = 1.5            # 单个轮子质量 (kg)

    # 轮子位置
    wheel_track: float = 0.3420        # 轮距 - 左右轮中心距 (m) - ROS配置
    wheel_x_offset: float = 0.0        # 轮子X轴偏移 (m)
    wheel_z_offset: float = -0.0805    # 轮子Z轴偏移 (m)

    # 万向轮
    caster_radius: float = 0.03        # 万向轮半径 (m)
    caster_mass: float = 0.1           # 万向轮质量 (kg)

    @property
    def wheel_radius(self) -> float:
        """获取轮子半径（米）"""
        return self.wheel_diameter / 2.0

    @property
    def half_track_width(self) -> float:
        """获取半轮距（米）"""
        return self.wheel_track / 2.0


# =============================================================================================
# 执行器（电机）参数
# =============================================================================================

@dataclass
class DashGoActuatorSpecs:
    """
    DashGo 执行器参数

    来源: Isaac Lab仿真配置 + ROS配置
    """
    # 驱动系统
    drive_type: str = "differential_drive"  # 驱动类型：差速驱动
    motor_type: str = "brushed_encoder"     # 电机类型：有刷编码马达

    # 编码器参数
    encoder_resolution: int = 1200         # 编码器线数 (ticks/转)
    gear_ratio: float = 1.0                # 减速比
    motor_reverse: bool = False            # 电机反转

    # Isaac Lab仿真参数
    stiffness: float = 0.0                 # 刚度系数 (N·m/rad) - 速度控制模式
    damping: float = 5.0                   # 阻尼系数 (N·m·s/rad)
    effort_limit: float = 20.0             # 力矩上限 (N·m)
    velocity_limit: float = 5.0            # 速度上限 (rad/s)

    # ROS PID控制参数
    kp: float = 50.0                       # 比例增益
    kd: float = 20.0                       # 微分增益
    ki: float = 0.0                        # 积分增益（未使用）
    ko: float = 50.0                       # 前馈增益

    @property
    def max_wheel_angular_velocity(self) -> float:
        """计算最大轮子角速度 (rad/s)"""
        # 从最大线速度0.3 m/s计算
        max_lin_vel = MOTION_CONFIG["max_lin_vel"]
        return max_lin_vel / (0.1264 / 2.0)  # v / r


# =============================================================================================
# 运动参数
# =============================================================================================

@dataclass
class DashGoMotionSpecs:
    """
    DashGo 运动参数

    来源: ROS导航配置 (base_local_planner_params.yaml)
    """
    # 速度限制
    max_lin_vel: float = 0.3              # 最大线速度 (m/s)
    min_lin_vel: float = -0.3             # 最小线速度 (m/s) - 倒车
    max_ang_vel: float = 1.0              # 最大角速度 (rad/s)
    min_ang_vel: float = -1.0             # 最小角速度 (rad/s)
    min_in_place_vel: float = 0.4         # 原地旋转速度 (rad/s)

    # 加速度限制
    max_lin_acc: float = 1.0              # 最大线加速度 (m/s²)
    max_ang_acc: float = 0.6              # 最大角加速度 (rad/s²)

    # 控制频率
    ros_control_rate: float = 10.0        # ROS base_controller频率 (Hz)
    ros_serial_rate: float = 50.0         # ROS serial_rate (Hz)
    isaac_control_dt: float = 0.05        # Isaac Lab控制周期 (s，当前按 20Hz 对齐实车)
    isaac_sim_dt: float = 0.005           # Isaac Lab仿真周期 (s)

    # 物理仿真参数
    linear_damping: float = 0.1           # 线性阻尼
    angular_damping: float = 0.1          # 角阻尼
    max_phys_lin_vel: float = 10.0        # 物理上线速度上限 (m/s)
    max_phys_ang_vel: float = 10.0        # 物理上角速度上限 (rad/s)

    @property
    def max_rotation_per_second(self) -> float:
        """获取每秒最大旋转圈数"""
        return self.max_ang_vel / (2.0 * 3.14159)


# =============================================================================================
# 传感器参数
# =============================================================================================

@dataclass
class DashGoLidarSpecs:
    """
    DashGo LiDAR传感器规格

    型号: Lakibeam 单雷达前向扇区配置
    来源: drivers/lakibeam_driver + ROS2 实车配置
    """
    # 基本信息
    model: str = "Lakibeam Front Sector"  # 型号
    alias: str = "lakibeam1"              # 驱动包名

    # 扫描参数
    scan_fov: float = 180.0               # 扫描视野 (度)
    max_range_real: float = 12.0          # 实机部署使用的有效距离上限 (m)
    max_range_sim: float = 12.0           # 仿真最大距离 (m) - RayCaster
    min_range: float = 0.15               # 最小距离 (m)

    # 频率参数
    scan_frequency_min: float = 10.0      # 最小扫描频率 (Hz)
    scan_frequency_max: float = 30.0      # 最大扫描频率 (Hz)
    scan_frequency_default: float = 30.0  # 默认扫描频率 (Hz)
    sample_rate: float = 43200.0          # 约 1440 点 * 30Hz

    # 数据输出
    data_points_per_scan: int = 1440      # 实机 /scan 常见点数
    sim_channels_v6: int = 216            # 仿真原始射线数（两台108宽相机拼接）
    angular_resolution: float = 0.125     # 实机角度分辨率近似值 (度)

    # 仿真配置
    sim_update_rate: float = 10.0         # 仿真更新频率 (Hz)
    sim_install_height: float = 0.13      # 安装高度 (m) - 对齐实物
    sim_num_sectors: int = 72             # 策略输入维度保持72
    sim_horizontal_res: float = 180.0 / 216.0  # 仿真水平分辨率 (度)

    # 障碍物检测
    obstacle_detection_range: float = 3.0    # 障碍物检测距离 (m) - ROS配置
    obstacle_clearing_range: float = 3.5     # 障碍物清除距离 (m) - ROS配置

    @property
    def sector_angle(self) -> float:
        """获取每个扇区的角度 (度)"""
        return self.scan_fov / self.sim_num_sectors

    @property
    def points_per_sector_real(self) -> int:
        """实物每个扇区的点数"""
        return self.data_points_per_scan // self.sim_num_sectors  # 1440 // 72 = 20

    @property
    def points_per_sector_sim(self) -> int:
        """仿真每个扇区的点数（v6.0）"""
        return self.sim_channels_v6 // self.sim_num_sectors  # 216 // 72 = 3


# =============================================================================================
# 通信参数
# =============================================================================================

@dataclass
class DashGoCommSpecs:
    """
    DashGo 通信参数

    来源: ROS配置
    """
    port: str = "/dev/ttyUSB0"            # 串口设备
    port_alt: str = "/dev/dashgo"         # 备用串口
    baud_rate: int = 115200               # 波特率 (bps)
    timeout: float = 0.1                  # 超时时间 (s)
    rate: float = 50.0                    # 通信频率 (Hz)
    sensor_state_rate: float = 10.0       # 传感器状态频率 (Hz)


# =============================================================================================
# 电源系统参数
# =============================================================================================

@dataclass
class DashGoPowerSpecs:
    """
    DashGo 电源系统参数

    来源: 官方规格 + 估算
    """
    nominal_voltage: float = 12.0         # 标称电压 (V)
    battery_capacity: float = 10.0        # 电池容量 (Ah) - 典型配置
    motor_power_peak: float = 20.0        # 电机峰值功率 (W)

    # 续航估算
    endurance_time_flat: float = 2.0      # 平坦地面续航 (小时)
    endurance_distance: float = 2.0       # 续航里程 (km) - 以0.3m/s速度

    @property
    def battery_energy(self) -> float:
        """电池总能量 (Wh)"""
        return self.nominal_voltage * self.battery_capacity


# =============================================================================================
# 导航参数
# =============================================================================================

@dataclass
class DashGoNavigationSpecs:
    """
    DashGo 导航参数

    来源: ROS导航配置
    """
    # 目标公差
    xy_goal_tolerance: float = 0.2        # 位置公差 (m)
    yaw_goal_tolerance: float = 0.1       # 姿态公差 (rad)

    # 路径规划
    sim_time: float = 0.6                 # 前瞻时间 (s)
    vx_samples: int = 12                  # 线速度采样数
    vtheta_samples: int = 15              # 角速度采样数


# =============================================================================================
# 完整机器人配置类
# =============================================================================================

@dataclass
class DashGoRobotConfig:
    """
    DashGo D1 机器人完整配置

    整合所有参数类别，提供统一的访问接口。
    """
    physical: DashGoPhysicalSpecs = field(default_factory=DashGoPhysicalSpecs)
    actuator: DashGoActuatorSpecs = field(default_factory=DashGoActuatorSpecs)
    motion: DashGoMotionSpecs = field(default_factory=DashGoMotionSpecs)
    lidar: DashGoLidarSpecs = field(default_factory=DashGoLidarSpecs)
    comm: DashGoCommSpecs = field(default_factory=DashGoCommSpecs)
    power: DashGoPowerSpecs = field(default_factory=DashGoPowerSpecs)
    navigation: DashGoNavigationSpecs = field(default_factory=DashGoNavigationSpecs)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'DashGoRobotConfig':
        """
        从ROS YAML配置文件加载参数

        Args:
            yaml_path: YAML文件路径。如果为None，使用默认路径。

        Returns:
            DashGoRobotConfig: 完整配置对象
        """
        if yaml_path is None:
            yaml_path = str(EAI_PARAMS_YAML)

        if not os.path.exists(yaml_path):
            print(f"[DashGoRobotConfig] YAML文件不存在: {yaml_path}，使用默认值")
            return cls()

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                params = _unwrap_ros_parameter_payload(yaml.safe_load(f))

            # 提取物理参数
            physical = DashGoPhysicalSpecs(
                wheel_diameter=params.get("wheel_diameter", DashGoPhysicalSpecs().wheel_diameter),
                wheel_track=params.get("wheel_track", DashGoPhysicalSpecs().wheel_track),
            )

            # 提取执行器参数
            actuator = DashGoActuatorSpecs(
                encoder_resolution=params.get("encoder_resolution", DashGoActuatorSpecs().encoder_resolution),
            )

            # 提取PID参数
            if "Kp" in params:
                actuator.kp = params["Kp"]
            if "Kd" in params:
                actuator.kd = params["Kd"]
            if "Ki" in params:
                actuator.ki = params["Ki"]
            if "Ko" in params:
                actuator.ko = params["Ko"]

            print(f"[DashGoRobotConfig] 从YAML加载参数: {yaml_path}")
            print(f"[DashGoRobotConfig] wheel_diameter={physical.wheel_diameter}, "
                  f"wheel_track={physical.wheel_track}")

            return cls(
                physical=physical,
                actuator=actuator,
            )

        except Exception as e:
            print(f"[DashGoRobotConfig] 读取YAML文件失败: {e}，使用默认值")
            return cls()

    def summary(self) -> str:
        """生成配置摘要"""
        summary = f"""
{'='*80}
DashGo D1 机器人配置摘要
{'='*80}
物理尺寸:
  - 主体直径: {self.physical.body_diameter*100:.1f} cm
  - 主体高度: {self.physical.body_height*100:.1f} cm
  - 主体质量: {self.physical.body_mass} kg
  - 轮子直径: {self.physical.wheel_diameter*100:.1f} cm
  - 轮距: {self.physical.wheel_track*100:.1f} cm

运动参数:
  - 最大线速度: {self.motion.max_lin_vel} m/s
  - 最大角速度: {self.motion.max_ang_vel} rad/s
  - 最大线加速度: {self.motion.max_lin_acc} m/s²
  - 最大角加速度: {self.motion.max_ang_acc} rad/s²

LiDAR传感器 ({self.lidar.model}):
  - 扫描范围: {self.lidar.scan_fov}°
  - 最大距离: {self.lidar.max_range_real} m (实物), {self.lidar.max_range_sim} m (仿真)
  - 扫描频率: {self.lidar.scan_frequency_min}-{self.lidar.scan_frequency_max} Hz
  - 数据点数: {self.lidar.data_points_per_scan} 点/圈 (实物)
  - 仿真射线: {self.lidar.sim_channels_v6} 射线 (v6.0)

执行器:
  - 编码器: {self.actuator.encoder_resolution} ticks/转
  - PID: Kp={self.actuator.kp}, Kd={self.actuator.kd}, Ki={self.actuator.ki}, Ko={self.actuator.ko}

通信:
  - 串口: {self.comm.port}
  - 波特率: {self.comm.baud_rate} bps
  - 频率: {self.comm.rate} Hz

电源:
  - 电压: {self.power.nominal_voltage} V
  - 容量: {self.power.battery_capacity} Ah
  - 续航: {self.power.endurance_time_flat} 小时 (平坦地面)

{'='*80}
"""
        return summary


# =============================================================================================
# 向后兼容的别名类
# =============================================================================================

class DashGoROSParams(DashGoRobotConfig):
    """
    向后兼容的别名类

    保持与旧代码的兼容性。
    """
    @property
    def wheel_diameter(self) -> float:
        return self.physical.wheel_diameter

    @property
    def wheel_radius(self) -> float:
        return self.physical.wheel_radius

    @property
    def wheel_track(self) -> float:
        return self.physical.wheel_track

    @property
    def encoder_resolution(self) -> int:
        return self.actuator.encoder_resolution


# =============================================================================================
# 测试代码
# =============================================================================================

if __name__ == "__main__":
    # 测试参数加载
    print("="*80)
    print("DashGo D1 机器人配置测试")
    print("="*80)

    # 从YAML加载（如果存在）
    config = DashGoRobotConfig.from_yaml()

    # 打印完整摘要
    print(config.summary())

    # 测试向后兼容类
    print("\n向后兼容测试:")
    params = DashGoROSParams.from_yaml()
    print(f"  轮子直径: {params.wheel_diameter} m")
    print(f"  轮子半径: {params.wheel_radius} m")
    print(f"  轮距: {params.wheel_track} m")
    print(f"  编码器: {params.encoder_resolution} ticks/转")

    # 测试常量配置
    print("\n常量配置测试:")
    print(f"  MOTION_CONFIG: {MOTION_CONFIG}")
    print(f"  LIDAR_CONFIG: {LIDAR_CONFIG}")
    print(f"  NAVIGATION_CONFIG: {NAVIGATION_CONFIG}")
