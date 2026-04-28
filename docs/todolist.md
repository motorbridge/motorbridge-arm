# Arm SDK 机械臂层能力分析与实现 ToDo（基于 motorbridge Python binding）

更新时间：2026-04-27
位置：`/home/w0x7ce/Downloads/dm_candrive/arm_sdk`

## 1. 现有 3 套 SDK 的“机械臂层”能力对比（结论）

> 目标不是比较电机驱动，而是比较“面向整机机械臂”的 SDK 能力。

| 能力域 | arx5-sdk | piper_sdk | z1_sdk | 我们建议的 reBot SDK 目标 |
|---|---|---|---|---|
| 连接/会话管理 | 有（接口名、模型、控制器实例） | 有（CAN 接口管理） | 有（ArmInterface 生命周期） | 必须有 |
| 关节状态读取 | 有 | 有 | 有 | 必须有 |
| 末端状态（笛卡尔） | 有 | 有（通过接口） | 有 | 必须有 |
| 运动模式管理 | 有（joint/cartesian） | 有（多模式） | 有（FSM） | 必须有 |
| MoveJ | 有 | 有 | 有 | 必须有 |
| MoveL | 有 | 有 | 有 | 必须有 |
| MoveC | 有（轨迹/笛卡尔） | 有 | 有 | 必须有（可后置优化） |
| 使能/失能/急停 | 有 | 有 | 有 | 必须有 |
| 回零/零位标定 | 有 | 有 | 有 | 必须有 |
| 参数读写（电机/系统） | 有（部分） | 很强 | 中等 | 必须有 |
| 安全限制（限位/限速/限扭） | 有且强调 | 有 | 有 | 必须有（高优先级） |
| FK/IK/雅可比/逆动力学 | 有 | 部分 | 有 | 建议有（先 FK/IK） |
| 示教/复现 | 有 | 有（主从/示教相关） | 有 | 规划支持 |
| 多臂/主从 | 有（多线程） | 有（主从配置） | 示例有 | 规划支持 |
| gripper 专项接口 | 有 | 有 | 有 | 必须有 |
| 高层任务编排 | 较弱 | 中 | 中 | 我们要补齐 |

### 1.1 核心观察

1. 这三套 SDK 的共同点：都不仅是“单电机命令封装”，而是“机械臂对象 + 运动语义 + 安全策略 + 标定参数”。
2. 你当前的 `motorbridge` 已经有跨厂商电机原语（enable/move/mode/param/scan），这正好适合作为机械臂 SDK 底座。
3. 目前缺口主要在“机械臂层抽象”：关节组建模、运动学、轨迹执行器、安全监督器、校准流程状态机。

### 1.2 MoveJ / MoveL / MoveC 是什么（功能解释）

1. `MoveJ`（Joint Space Motion，关节空间运动）
   - 含义：直接给各关节目标角度 `q_target`，机械臂按关节轨迹到位。
   - 特点：实现简单、鲁棒性高、速度快；末端路径通常不是直线。
   - 常见用途：大幅姿态切换、回零、点到点搬运。

2. `MoveL`（Linear Cartesian Motion，笛卡尔直线运动）
   - 含义：让末端执行器 TCP 在笛卡尔空间沿直线移动到目标位姿。
   - 特点：工艺路径可控，但需要插补 + 反解（IK），计算更复杂。
   - 常见用途：插拔、点胶、焊接、直线接近工件。

3. `MoveC`（Circular Motion，圆弧/圆周运动）
   - 含义：让 TCP 沿圆弧轨迹运动（常见为“中间点 + 终点”定义圆弧）。
   - 特点：比 MoveL 更复杂，需要圆弧几何插补 + 连续 IK。
   - 常见用途：打磨圆角、圆弧涂胶、曲线避障。

### 1.3 FK / IK 是什么（为什么要做）

1. `FK`（Forward Kinematics，正向运动学）
   - 输入：关节角 `q`。
   - 输出：末端位姿 `T(q)`（位置 + 姿态）。
   - 用途：把电机反馈角度转成“机械臂末端在哪里”，用于 UI 显示、误差评估、碰撞/工作空间判断。

2. `IK`（Inverse Kinematics，逆向运动学）
   - 输入：目标末端位姿 `T_target`。
   - 输出：一组可行关节角 `q_target`。
   - 用途：MoveL/MoveC 必需能力；用户给末端路径，系统转成每关节命令。

3. 在本项目中 FK/IK 的落地关系
   - `MoveJ`：可不依赖 IK（直接关节轨迹）。
   - `MoveL/MoveC`：依赖 IK（每个插补点都要解一组 `q`）。
   - 状态展示：依赖 FK（把 `motor.get_state().pos` 映射为 TCP 位姿）。

---

## 2. 基于 motorbridge 的可复用底层能力（已具备）

从 `motorbridge/bindings/python/src/motorbridge/core.py` 可直接复用：

- Controller 级：
  - `enable_all()` / `disable_all()`
  - `add_damiao_motor()` / `add_robstride_motor()` / ...
  - `poll_feedback_once()`
- Motor 级：
  - 控制：`enable()` `disable()` `clear_error()`
  - 模式：`ensure_mode(...)`
  - 命令：`send_mit()` `send_pos_vel()` `send_vel()` `send_force_pos()`
  - 状态：`get_state()` `request_feedback()`
  - 零点：`set_zero_position()`
  - 参数：`*_get_param_*`, `*_write_param_*`, `write_register_*`, `get_register_*`
  - RobStride 特有：`robstride_ping()`, `robstride_set_device_id()`, typed param read/write

结论：底层“驱动原语”基本齐全，可支撑机械臂层封装。

---

## 3. reBot 机械臂 SDK 目标能力（产品级）

## 3.1 API 视角（必须有）

1. `Arm.connect()` / `Arm.close()`
2. `Arm.enable()` / `Arm.disable()` / `Arm.estop()`
3. `Arm.get_joint_state()` / `Arm.get_pose()`
4. `Arm.move_j(q, v, a)`
5. `Arm.move_l(pose, v, a)`
6. `Arm.move_c(mid, end, v, a)`
7. `Arm.home()` / `Arm.zero_calibrate()`
8. `Arm.read_param(scope=joint/system)` / `Arm.write_param(...)`
9. `Arm.set_tool(...)` / `Arm.set_payload(...)`
10. `Arm.get_faults()` / `Arm.clear_faults()`

## 3.2 工程能力（必须有）

1. 统一模型描述：DOF、关节方向、gear ratio、软硬限位、默认 PID/阻尼。
2. 安全监督器：输入命令门控、速度/位置/力矩钳位、超时降级、异常停机。
3. 轨迹执行器：时间参数化（梯形/S 曲线）+ 采样下发。
4. 状态机：`DISCONNECTED -> IDLE -> ENABLED -> RUNNING -> FAULT`。
5. 校准流程：单关节零位、全臂回零、校准结果持久化。
6. 日志与回放：命令/反馈时序记录，便于复盘。

## 3.3 扩展能力（第二阶段）

1. teach/replay
2. 双臂同步/主从
3. 轨迹文件导入导出（CSV/JSON）
4. ROS2 adapter

---

## 4. 建议代码结构（新 SDK）

建议新目录：`/home/w0x7ce/Downloads/dm_candrive/arm_sdk/rebot_sdk`（后续创建）

```text
rebot_sdk/
  rebot_sdk/
    __init__.py
    arm.py                  # Arm 主对象
    session.py              # motorbridge controller/motor 生命周期
    model/
      profiles.py           # 各机型关节拓扑和限制
      kinematics.py         # FK/IK（先可接 pinocchio 或简化解）
    motion/
      planner.py            # moveJ/moveL/moveC 轨迹生成
      executor.py           # 轨迹下发与实时循环
    safety/
      supervisor.py         # 限位限速限扭、异常处理
    calibration/
      zeroing.py            # 零位流程
    params/
      registry.py           # 参数字典（按 vendor/type）
    telemetry/
      state_cache.py        # 反馈聚合
      recorder.py           # 时序记录
  examples/
  tests/
  pyproject.toml
```

---

## 5. 实施清单（按优先级排序，可直接开工）

### 5.1 里程碑 A：规格与数据模型定稿

目标：先把接口、数据结构、机型配置固定，避免后续反复返工。

实施步骤：
- [ ] 新建 `spec/spec.md`，定义最小 API：`connect/enable/disable/get_state/move_j/home/estop`
- [ ] 定义 `ArmConfig`：`vendor/model/dof/joint_map/feedback_map/joint_dir/zero_offset/limit_pos/limit_vel/limit_tau`
- [ ] 定义 `ArmState`、`JointState`、`FaultState`、`CommandResult`
- [ ] 定义统一错误码：`ERR_TIMEOUT/ERR_MODE/ERR_LIMIT/ERR_SINGULAR/ERR_NO_IK/ERR_BUS`
- [ ] 定义参数元数据结构：`param_id/type/access/default/unit/desc/vendor`

交付物：
- `spec/spec.md`
- `spec/api.md`
- `spec/error_codes.md`

验收标准：
- [ ] 团队按文档可独立实现，不需要口头补充规则

### 5.2 里程碑 B：通信会话层与设备编排

目标：把 `motorbridge` 原语封装成“机械臂会话”，屏蔽底层细节。

实施步骤：
- [ ] 实现 `session.py`：`MotorBridgeSession`（controller 生命周期、handle 注册、关闭顺序）
- [ ] 提供统一 attach 流程：`add_joint(vendor, esc_id, feedback_id, model)`
- [ ] 实现在线检查：连接后自动 `request_feedback + get_state`
- [ ] 实现基础批量操作：`enable_all/disable_all/clear_fault_all`
- [ ] 增加会话级日志：连接参数、各关节注册信息、失败原因

交付物：
- `rebot_sdk/session.py`
- `examples/00_session_smoke.py`

验收标准：
- [ ] 可稳定完成 `connect -> attach joints -> enable -> disable -> close`

### 5.3 里程碑 C：最小机械臂控制闭环（MVP）

目标：先跑通单臂基本控制链路。

实施步骤：
- [ ] 实现 `arm.py` 主类，暴露 `Arm.connect/enable/disable/get_joint_state/move_j/home`
- [ ] 实现状态缓存 `state_cache.py`（轮询更新 + 时间戳）
- [ ] 实现 `move_j` 规划与执行：关节线性插值 + 周期下发 `send_pos_vel`
- [ ] 实现 `home`：按机型默认姿态运动并收敛判定
- [ ] 增加示例：`examples/01_connect_enable_movej.py`

交付物：
- `rebot_sdk/arm.py`
- `rebot_sdk/telemetry/state_cache.py`
- `rebot_sdk/motion/executor.py`

验收标准：
- [ ] 单臂能稳定执行 `connect -> enable -> move_j -> home -> disable`

### 5.4 里程碑 D：安全监督与故障处理

目标：避免越界、失控、通信异常导致风险动作。

实施步骤：
- [ ] 实现 `safety/supervisor.py`：位置/速度/力矩钳位
- [ ] 接入 watchdog：控制循环超过阈值自动 `hold/stop`
- [ ] 异常分级：可恢复（重试）/不可恢复（进入 FAULT）
- [ ] 增加命令去抖与重复命令合并
- [ ] 完成故障处理动作：`estop -> disable -> fault_report`

交付物：
- `rebot_sdk/safety/supervisor.py`
- `tests/test_safety_limits.py`
- `tests/test_watchdog_timeout.py`

验收标准：
- [ ] 断线、超时、越界场景下动作可控且可恢复

### 5.5 里程碑 E：参数系统与零位校准

目标：完成参数读写闭环与可重复校准流程。

实施步骤：
- [ ] 实现参数字典层：`params/registry.py`（按 vendor + param_id + type + access）
- [ ] 对接 RobStride typed param：`i8/u8/u16/u32/f32` 读写与回读校验
- [ ] 对接 Damiao register/param：`u32/f32` 读写与回读校验
- [ ] 实现 `zero_calibrate(joint|all)`：前置检查 -> 执行 -> 二次确认 -> 结果记录
- [ ] 实现参数持久化：设备 `store_parameters` + 本地 `snapshot.json`

交付物：
- `rebot_sdk/params/registry.py`
- `rebot_sdk/calibration/zeroing.py`
- `examples/02_param_rw_and_zero.py`

验收标准：
- [ ] 关键参数可读可写可回读一致
- [ ] 零位流程可重复执行且结果可追踪

### 5.6 里程碑 F：运动学与轨迹功能（MoveL/MoveC）

目标：补齐机械臂层核心能力，不只停留在 MoveJ。

实施步骤：
- [ ] 在 `model/profiles.py` 固化 DH 或 URDF 映射参数
- [ ] 实现 FK：`forward(q) -> pose`
- [ ] 实现 IK：`inverse(T_target, q_seed) -> q_target`（数值法起步）
- [ ] 实现 `move_l`：笛卡尔直线插补 -> IK -> `send_pos_vel`
- [ ] 实现 `move_c`：圆弧插补 -> IK -> `send_pos_vel`
- [ ] 增加无解/奇异处理：降速、回退、拒绝执行

交付物：
- `rebot_sdk/model/kinematics.py`
- `rebot_sdk/motion/planner.py`
- `examples/03_movel_movec.py`

验收标准：
- [ ] MoveL 直线偏差可量化并在阈值内
- [ ] MoveC 轨迹连续、无明显速度突变

### 5.7 里程碑 G：夹爪、工具与负载模型

目标：补齐工业可用功能，不仅控制关节。

实施步骤：
- [ ] 增加 `set_tool` / `set_payload` 接口
- [ ] 增加 gripper 控制统一接口（位置/速度/力）
- [ ] 在 FK/IK 中引入工具坐标系变换
- [ ] 更新安全阈值（负载变化影响速度/力矩限制）

交付物：
- `rebot_sdk/tooling.py`
- `examples/04_gripper_and_tool.py`

验收标准：
- [ ] 更换工具后末端位姿与运动行为仍可预测

### 5.8 里程碑 H：测试、文档与发布

目标：形成可发布、可维护的产品级 SDK。

实施步骤：
- [ ] 单元测试：状态机、参数解析、限位钳位、IK 边界
- [ ] HIL 测试脚本：连接、扫描、使能、MoveJ/L/C、零位、异常恢复
- [ ] 文档：快速开始、API 参考、参数字典、故障排查
- [ ] 版本发布：`pyproject.toml`、changelog、tag 规范
- [ ] 对齐 `motorbridge-studio`：共享错误码与参数命名

交付物：
- `tests/` 全套测试
- `docs/` 文档
- 发布脚本与版本说明

验收标准：
- [ ] 新用户按文档可独立完成从安装到动作执行的全流程

---

## 6. 与 motorbridge 的接口映射（落地表）

| reBot SDK 动作 | motorbridge 调用建议 |
|---|---|
| connect | `Controller(channel)` + `add_*_motor(...)` |
| enable/disable | `enable_all()` / `disable_all()` 或逐电机 `enable()/disable()` |
| move_j | `ensure_mode(pos-vel)` + `send_pos_vel()` 周期下发 |
| compliant/mit | `ensure_mode(mit)` + `send_mit()` |
| stop/hold | `send_vel(0)` 或切安全模式 + disable |
| read state | `request_feedback()` + `get_state()` |
| set zero | `disable()` -> `set_zero_position()` -> `enable()` |
| set id | RobStride: `robstride_set_device_id()` |
| param rw | RobStride typed `get/write_param_*`, Damiao `get/write_param_*` |

---

## 7. 关键风险与先行约束

1. 不同厂商模式切换语义不一致：必须有 vendor profile 与 capability flags。
2. Feedback ID / responder ID 机制差异：机械臂层不能假设“每关节唯一 responder”。
3. 实时性与 Python GIL：控制环尽量简化，必要时将高频环下沉到 Rust/C++。
4. MoveL/MoveC 的 IK 收敛性：先限定工作空间和速度，再逐步开放。
5. 零位标定有风险：必须做“二次确认 + 速度阈值 + 电流阈值”保护。

---

## 8. 下一步执行建议（按你当前项目最实用）

1. 先做 Phase 0 + Phase 1（MVP）并接入你现有 `motorbridge` 环境。
2. 先只支持 `rebot-arm-robstride` 单机型，减少变量。
3. 等 MVP 稳定后再扩展 MoveL/MoveC 与多臂。
