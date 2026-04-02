# Phase 2 实机边界与接口差异

创建时间: 2026-04-02

## 结论

- 第二阶段的目标不是把 upstream `NavRL` 的 ROS2 感知/安全包原样搬到 DashGo 上。
- 第二阶段的正确目标是：在 DashGo 当前 `LaserScan + odom + goal + global plan + cmd_vel` 合同下，设计激光雷达/差速友好的动态障碍表示与安全动作接口。
- upstream `onboard_detector + safe_action` 只作为参考实现，不是直接复用目标。

## 为什么不能直接复用 upstream ROS2 栈

- upstream `navigation_runner` 依赖以下服务：
  - `/occupancy_map/raycast`
  - `/onboard_detector/get_dynamic_obstacles`
  - `/safe_action/get_safe_action`
- upstream `onboard_detector` 默认输入是：
  - `depth_image_topic`
  - `color_image_topic`
  - `YOLO detection`
  - `pose/odom`
  - 相机内参、深度尺度、点云聚类和 tracking 参数
- upstream `safe_action` 也不是简单限幅，而是基于动态障碍 token 和激光打点做 ORCA 风格安全速度求解。
- 当前 DashGo 主线真实合同是：
  - `/scan`
  - `/odom`
  - `/goal_pose`
  - `/dashgo/global_plan`
  - `DynamicsSafetyFilter`
- 因此两边的差异不在“有没有动态障碍模块”，而在：
  - 传感器前提不同
  - 消息与 service 合同不同
  - 机器人运动学不同

## 第二阶段默认范围

- 保留 DashGo 差速动作语义，不引入上游全向 `linear.x + linear.y + angular.z` 控制接口。
- 保留 DashGo 180 度激光雷达主线，不把深度相机链作为默认前提。
- 为 DashGo 定义自己的动态障碍中间表示：
  - 基于激光雷达的局部动态障碍 position/velocity/size token
  - 或等价的激光雷达可消费风险表示
- 为 DashGo 定义自己的安全接口：
  - 保留现有 `DynamicsSafetyFilter` 路线，或
  - 新增与差速动作兼容的安全动作修正层
- 若以后明确新增深度相机和检测链，再讨论是否引入更贴近 upstream 的 detector 分支

## 第二阶段不纳入默认目标的内容

- 不默认接入 upstream 的 `depth + color + yolo` 感知链
- 不默认要求 ROS2 侧完整复刻 upstream service graph
- 不默认把第一阶段候选 checkpoint 直接拿去实机部署
- 不把“原样迁移 upstream ROS2 包”当作成功标准

## 第二阶段建议顺序

- 先做 DashGo 动态障碍表示设计
- 再做 DashGo 安全动作接口设计
- 再做实机节点接线与离线回放验证
- 最后才做上车闭环测试

## 当前可复用与不可复用边界

- 可直接复用：
  - 第一阶段已经落地的 NavRL-style 训练栈
  - DashGo official-route 仿真环境
  - quick/main 正式评测协议
  - 对比工具与文档模板
- 不能直接复用：
  - upstream `onboard_detector` 的传感器输入合同
  - upstream `safe_action` 的服务接线方式
  - upstream 全向动作接口
