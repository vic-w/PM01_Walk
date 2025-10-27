"""自定义观测项与步态时钟实现。"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Optional

import torch


DEFAULT_STEP_CLOCK_JOINT_PARAMETERS = [
    {"joint_names": ["j00_hip_pitch_l"], "amplitude": 0.35, "phase_offset": 0.0},
    {"joint_names": ["j01_hip_roll_l"], "amplitude": 0.12, "phase_offset": 0.5 * math.pi},
    {"joint_names": ["j02_hip_yaw_l"], "amplitude": 0.08, "phase_offset": 0.0},
    {"joint_names": ["j03_knee_pitch_l"], "amplitude": 0.65, "phase_offset": 0.5 * math.pi},
    {"joint_names": ["j04_ankle_pitch_l"], "amplitude": 0.35, "phase_offset": -0.5 * math.pi},
    {"joint_names": ["j05_ankle_roll_l"], "amplitude": 0.12, "phase_offset": 0.5 * math.pi},
    {"joint_names": ["j06_hip_pitch_r"], "amplitude": 0.35, "phase_offset": math.pi},
    {"joint_names": ["j07_hip_roll_r"], "amplitude": 0.12, "phase_offset": 0.5 * math.pi + math.pi},
    {"joint_names": ["j08_hip_yaw_r"], "amplitude": 0.08, "phase_offset": math.pi},
    {"joint_names": ["j09_knee_pitch_r"], "amplitude": 0.65, "phase_offset": 0.5 * math.pi + math.pi},
    {"joint_names": ["j10_ankle_pitch_r"], "amplitude": 0.35, "phase_offset": -0.5 * math.pi + math.pi},
    {"joint_names": ["j11_ankle_roll_r"], "amplitude": 0.12, "phase_offset": 0.5 * math.pi + math.pi},
    {"joint_names": ["j12_waist_yaw"], "amplitude": 0.08, "phase_offset": 0.0},
    {"joint_names": ["j13_shoulder_pitch_l"], "amplitude": 0.35, "phase_offset": math.pi},
    {"joint_names": ["j14_shoulder_roll_l"], "amplitude": 0.2, "phase_offset": math.pi},
    {"joint_names": ["j15_shoulder_yaw_l"], "amplitude": 0.2, "phase_offset": math.pi},
    {"joint_names": ["j16_elbow_pitch_l"], "amplitude": 0.45, "phase_offset": math.pi},
    {"joint_names": ["j17_elbow_yaw_l"], "amplitude": 0.15, "phase_offset": math.pi},
    {"joint_names": ["j18_shoulder_pitch_r"], "amplitude": 0.35, "phase_offset": 0.0},
    {"joint_names": ["j19_shoulder_roll_r"], "amplitude": 0.2, "phase_offset": 0.0},
    {"joint_names": ["j20_shoulder_yaw_r"], "amplitude": 0.2, "phase_offset": 0.0},
    {"joint_names": ["j21_elbow_pitch_r"], "amplitude": 0.45, "phase_offset": 0.0},
    {"joint_names": ["j22_elbow_yaw_r"], "amplitude": 0.15, "phase_offset": 0.0},
    {"joint_names": ["j23_head_yaw"], "amplitude": 0.05, "phase_offset": 0.0},
]


def _get_device(env: Any) -> torch.device:
    device = getattr(env, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device


def _zeros(env: Any, dim: int) -> torch.Tensor:
    return torch.zeros((getattr(env, "num_envs", 1), dim), device=_get_device(env))


def _get_robot_data(env: Any) -> Any:
    """尝试获取机器人 Articulation 的数据对象。"""

    robot = None
    scene = getattr(env, "scene", None)
    if scene is not None:
        if hasattr(scene, "__getitem__"):
            try:
                robot = scene["robot"]
            except Exception:  # noqa: BLE001
                robot = None
        if robot is None:
            robot = getattr(scene, "robot", None)
    if robot is None:
        robot = getattr(env, "robot", None)
    data = getattr(robot, "data", None)
    return data


def _get_joint_names(data: Any, num_joints: int) -> list[str]:
    joint_names = getattr(data, "joint_names", None)
    if isinstance(joint_names, torch.Tensor):
        try:
            joint_names = [str(name) for name in joint_names]
        except Exception:  # noqa: BLE001
            joint_names = None
    if joint_names is None:
        return [""] * num_joints
    return list(joint_names)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.stack((quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]), dim=-1)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_vec = quat[..., 1:]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(quat), vec)


def _quaternion_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """将四元数转换为 XYZ 欧拉角。"""

    qw, qx, qy, qz = quat.unbind(-1)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1.0, torch.sign(sinp) * (torch.pi / 2.0), torch.asin(sinp))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


def _difference_with_reference(current: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
    if reference is None:
        return current * 0.0
    return current - reference


def _extract_frequency_range(step_params: Mapping[str, Any]) -> tuple[float, float]:
    freq_range = step_params.get("frequency_range")
    if freq_range is None:
        base_freq = float(step_params.get("base_frequency", 1.2))
        return base_freq, base_freq
    if isinstance(freq_range, (int, float)):
        value = float(freq_range)
        return value, value
    if len(freq_range) == 1:
        value = float(freq_range[0])
        return value, value
    freq_min = float(freq_range[0])
    freq_max = float(freq_range[1])
    if freq_min > freq_max:
        freq_min, freq_max = freq_max, freq_min
    return freq_min, freq_max


def _build_step_clock_profile(
    data: Any, step_params: Mapping[str, Any], num_joints: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    amplitude = torch.zeros(num_joints, device=device)
    phase_offset = torch.zeros(num_joints, device=device)
    amplitude_scale = float(step_params.get("amplitude_scale", 1.0))
    joint_params = step_params.get("joint_parameters")
    inherit_default = bool(step_params.get("inherit_default_parameters", True))
    parameters: list[Mapping[str, Any]] = []
    if joint_params is None or inherit_default:
        parameters.extend(DEFAULT_STEP_CLOCK_JOINT_PARAMETERS)
    if joint_params is not None:
        parameters.extend(joint_params)
    joint_names = _get_joint_names(data, num_joints)
    for entry in parameters:
        amplitude_value = float(entry.get("amplitude", 0.0)) * amplitude_scale
        phase_value = float(entry.get("phase_offset", 0.0))
        indices: set[int] = set()
        for name in entry.get("joint_names", []) or []:
            for idx, candidate in enumerate(joint_names):
                if candidate == name:
                    indices.add(idx)
        pattern = entry.get("pattern")
        if pattern:
            try:
                compiled = re.compile(pattern)
            except re.error:
                compiled = None
            if compiled is not None:
                for idx, candidate in enumerate(joint_names):
                    if compiled.search(candidate):
                        indices.add(idx)
        if not indices:
            continue
        for idx in indices:
            amplitude[idx] = amplitude_value
            phase_offset[idx] = phase_value
    return amplitude, phase_offset


def _compute_body_speed(data: Any, device: torch.device, num_envs: int) -> torch.Tensor:
    lin_vel = getattr(data, "root_lin_vel_w", None)
    if lin_vel is None:
        lin_vel = getattr(data, "root_lin_vel_b", None)
    if lin_vel is None:
        return torch.zeros(num_envs, device=device)
    if lin_vel.ndim == 1:
        lin_vel = lin_vel.unsqueeze(0)
    if lin_vel.shape[0] != num_envs:
        if lin_vel.shape[0] == 1:
            lin_vel = lin_vel.expand(num_envs, -1)
        else:
            lin_vel = lin_vel[:num_envs]
    return torch.linalg.norm(lin_vel[..., :2], dim=-1)


def _get_time_delta(env: Any, step_params: Mapping[str, Any]) -> float:
    if "dt" in step_params:
        return float(step_params["dt"])
    for attr in ("step_dt", "control_dt", "dt"):
        value = getattr(env, attr, None)
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        if isinstance(value, (int, float)):
            return float(value)
    sim = getattr(env, "sim", None)
    if sim is not None:
        dt_val = getattr(sim, "dt", None)
        if isinstance(dt_val, (int, float)):
            decimation = getattr(env, "decimation", 1)
            return float(dt_val) * float(decimation)
    return 0.02


def _ensure_step_clock_state(env: Any, data: Any, step_params: Mapping[str, Any]) -> Optional[dict[str, torch.Tensor]]:
    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        return None
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(0)
    device = joint_pos.device
    num_envs = joint_pos.shape[0]
    num_joints = joint_pos.shape[-1]
    state = getattr(env, "_pm01_step_clock_state", None)
    rebuild = True
    if isinstance(state, dict):
        amplitude = state.get("amplitude")
        phase_offset = state.get("phase_offset")
        phase = state.get("phase")
        frequency = state.get("frequency")
        if (
            isinstance(amplitude, torch.Tensor)
            and isinstance(phase_offset, torch.Tensor)
            and amplitude.shape[-1] == num_joints
            and phase_offset.shape[-1] == num_joints
            and isinstance(phase, torch.Tensor)
            and isinstance(frequency, torch.Tensor)
        ):
            rebuild = False
            state["amplitude"] = amplitude.to(device)
            state["phase_offset"] = phase_offset.to(device)
            state["phase"] = phase.to(device)
            state["frequency"] = frequency.to(device)
            if state["phase"].shape[0] != num_envs:
                state["phase"] = state["phase"][:1].expand(num_envs).clone()
            if state["frequency"].shape[0] != num_envs:
                state["frequency"] = state["frequency"][:1].expand(num_envs).clone()
        else:
            rebuild = True
    if rebuild:
        amplitude, phase_offset = _build_step_clock_profile(data, step_params, num_joints, device)
        phase = torch.rand(num_envs, device=device) * (2.0 * math.pi)
        freq_min, freq_max = _extract_frequency_range(step_params)
        if freq_max > freq_min:
            frequency = torch.empty(num_envs, device=device).uniform_(freq_min, freq_max)
        else:
            frequency = torch.full((num_envs,), freq_min, device=device)
        state = {
            "amplitude": amplitude,
            "phase_offset": phase_offset,
            "phase": phase,
            "frequency": frequency,
        }
        env._pm01_step_clock_state = state
    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor):
        env_ids = torch.nonzero(reset_buf, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            phase_noise = float(step_params.get("phase_noise", 0.0))
            state["phase"][env_ids] = torch.rand(env_ids.numel(), device=device) * (2.0 * math.pi)
            if phase_noise > 0.0:
                noise = torch.empty(env_ids.numel(), device=device).uniform_(-phase_noise, phase_noise)
                state["phase"][env_ids] = (state["phase"][env_ids] + noise) % (2.0 * math.pi)
            freq_min, freq_max = _extract_frequency_range(step_params)
            if freq_max > freq_min:
                state["frequency"][env_ids] = torch.empty(env_ids.numel(), device=device).uniform_(freq_min, freq_max)
            else:
                state["frequency"][env_ids] = freq_min
    return state


def _step_clock_update(
    env: Any,
    data: Any,
    step_params: Mapping[str, Any],
    update: bool,
) -> Optional[dict[str, torch.Tensor]]:
    state = _ensure_step_clock_state(env, data, step_params)
    if state is None:
        return None
    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        return None
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(0)
    device = joint_pos.device
    num_envs = joint_pos.shape[0]
    amplitude = state["amplitude"].unsqueeze(0).expand(num_envs, -1)
    phase_offset = state["phase_offset"].unsqueeze(0).expand(num_envs, -1)
    default_pos = getattr(data, "default_joint_pos", None)
    if default_pos is None:
        default_pos = torch.zeros_like(joint_pos)
    if default_pos.ndim == 1:
        default_pos = default_pos.unsqueeze(0).expand(num_envs, -1)
    phase = state["phase"]
    frequency = state["frequency"]
    if update:
        freq_limits = step_params.get("frequency_limits") or step_params.get("frequency_range")
        freq_min = freq_max = None
        if freq_limits is not None:
            if isinstance(freq_limits, (int, float)):
                freq_min = freq_max = float(freq_limits)
            else:
                freq_min = float(freq_limits[0])
                freq_max = float(freq_limits[1]) if len(freq_limits) > 1 else freq_min
                if freq_min > freq_max:
                    freq_min, freq_max = freq_max, freq_min
        speed_gain = float(step_params.get("speed_gain", 0.0))
        effective_freq = frequency
        if speed_gain != 0.0:
            body_speed = _compute_body_speed(data, device, num_envs)
            effective_freq = effective_freq + speed_gain * body_speed
        if freq_min is not None and freq_max is not None:
            effective_freq = torch.clamp(effective_freq, freq_min, freq_max)
        dt = _get_time_delta(env, step_params)
        phase = (phase + dt * effective_freq * (2.0 * math.pi)) % (2.0 * math.pi)
        state["phase"] = phase
        state["frequency_effective"] = effective_freq
    else:
        effective_freq = state.get("frequency_effective", frequency)
        if isinstance(effective_freq, torch.Tensor):
            effective_freq = effective_freq.to(device)
    reference = default_pos + amplitude * torch.sin(phase.unsqueeze(-1) + phase_offset)
    state["reference"] = reference
    state["frequency_effective"] = effective_freq
    env._pm01_step_clock_state = state
    env._pm01_joint_reference = reference
    env._pm01_step_phase = phase
    env._pm01_step_frequency = effective_freq
    env._pm01_step_frequency_base = frequency
    return state


def step_clock_state(
    env: Any, params: Optional[Mapping[str, Any]] = None, *, update: bool = False
) -> Optional[dict[str, torch.Tensor]]:
    step_params: Mapping[str, Any] = {} if params is None else params
    data = _get_robot_data(env)
    if data is None:
        return None
    return _step_clock_update(env, data, step_params, update)


def base_lin_vel_body_frame(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 3)

    if hasattr(data, "root_lin_vel_b"):
        return data.root_lin_vel_b

    lin_vel = getattr(data, "root_lin_vel_w", None)
    quat = getattr(data, "root_quat_w", None)
    if lin_vel is None or quat is None:
        return _zeros(env, 3)
    return _quat_rotate_inverse(quat, lin_vel)


def dof_pos_rel_to_default(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 0)

    joint_pos = getattr(data, "joint_pos", None)
    default_pos = getattr(data, "default_joint_pos", None)
    if joint_pos is None:
        return _zeros(env, 0)
    if default_pos is None:
        return joint_pos
    return joint_pos - default_pos


def dof_velocities(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 0)
    joint_vel = getattr(data, "joint_vel", None)
    if joint_vel is None:
        return _zeros(env, 0)
    return joint_vel


def previous_actions(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    action_name = None
    if params is not None:
        action_name = params.get("action_name")

    manager = getattr(env, "action_manager", None)
    if manager is None:
        return _zeros(env, 0)

    for attr in ("get_last_action", "get_previous_action", "get_previous_actions"):
        if hasattr(manager, attr):
            getter = getattr(manager, attr)
            try:
                if action_name is None:
                    action = getter()
                else:
                    action = getter(action_name)
            except TypeError:
                try:
                    action = getter(name=action_name)
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(action, torch.Tensor):
                return action
    if hasattr(manager, "last_action") and isinstance(manager.last_action, torch.Tensor):
        return manager.last_action
    if hasattr(manager, "data") and isinstance(manager.data, Mapping):
        data_map = manager.data.get(action_name or "joint_pos")
        if isinstance(data_map, torch.Tensor):
            return data_map
    return _zeros(env, 0)


def dof_pos_reference_diff(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 0)

    current = getattr(data, "joint_pos", None)
    reference = None
    if params is not None and params.get("use_step_clock", False):
        step_params = params.get("step_clock") or {}
        state = step_clock_state(env, step_params, update=True)
        if state is not None:
            reference = state.get("reference")
    if reference is None:
        command_manager = getattr(env, "command_manager", None)
        if command_manager is not None:
            for attr in ("get_reference_joint_pos", "joint_position_reference", "joint_pos_reference"):
                if hasattr(command_manager, attr):
                    reference = getattr(command_manager, attr)
                    if callable(reference):
                        try:
                            reference = reference()
                        except Exception:  # noqa: BLE001
                            reference = None
                    if isinstance(reference, torch.Tensor):
                        break
                    reference = None
    if current is None:
        return _zeros(env, 0)
    return _difference_with_reference(current, reference)


def base_ang_vel_body_frame(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 3)

    if hasattr(data, "root_ang_vel_b"):
        return data.root_ang_vel_b

    ang_vel = getattr(data, "root_ang_vel_w", None)
    quat = getattr(data, "root_quat_w", None)
    if ang_vel is None or quat is None:
        return _zeros(env, 3)
    return _quat_rotate_inverse(quat, ang_vel)


def base_euler_xyz(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 3)

    quat = getattr(data, "root_quat_w", None)
    if quat is None:
        return _zeros(env, 3)
    return _quaternion_to_euler_xyz(quat)


def random_push_force(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    dim = 3 if params is None else params.get("dim", 3)
    force_candidates: Iterable[torch.Tensor] = []
    if hasattr(env, "event_manager") and hasattr(env.event_manager, "data"):
        data_map = env.event_manager.data
        if isinstance(data_map, Mapping):
            candidate = data_map.get("rand_push_force")
            if isinstance(candidate, torch.Tensor):
                force_candidates = [candidate]
    for name in ("rand_push_force", "random_push_force", "external_force"):
        candidate = getattr(env, name, None)
        if isinstance(candidate, torch.Tensor):
            force_candidates = [candidate]
            break
    for tensor in force_candidates:
        if tensor.shape[-1] >= dim:
            return tensor[..., :dim]
    return _zeros(env, dim)


def random_push_torque(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    torque_candidates: Iterable[torch.Tensor] = []
    if hasattr(env, "event_manager") and hasattr(env.event_manager, "data"):
        data_map = env.event_manager.data
        if isinstance(data_map, Mapping):
            candidate = data_map.get("rand_push_torque")
            if isinstance(candidate, torch.Tensor):
                torque_candidates = [candidate]
    for name in ("rand_push_torque", "random_push_torque", "external_torque"):
        candidate = getattr(env, name, None)
        if isinstance(candidate, torch.Tensor):
            torque_candidates = [candidate]
            break
    for tensor in torque_candidates:
        if tensor.shape[-1] >= 3:
            return tensor[..., :3]
    return _zeros(env, 3)


def terrain_friction(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    friction = None
    terrain = getattr(env, "terrain", None)
    if terrain is not None:
        for attr in ("current_friction", "friction", "friction_coeff"):
            friction = getattr(terrain, attr, None)
            if isinstance(friction, torch.Tensor):
                break
    if friction is None and hasattr(env, "scene"):
        scene = env.scene
        friction = getattr(scene, "terrain_friction", None)
        if not isinstance(friction, torch.Tensor):
            friction = None
    if friction is None:
        return _zeros(env, 1)
    if friction.ndim == 0:
        friction = friction.expand(getattr(env, "num_envs", 1))
    return friction.reshape(getattr(env, "num_envs", 1), -1)[..., :1]


def body_mass(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 0)
    masses = getattr(data, "body_mass", None)
    if masses is None:
        return _zeros(env, 0)
    return masses


def stance_curve(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    num_legs = 2 if params is None else params.get("num_legs", 2)
    use_step_clock = False if params is None else params.get("use_step_clock", False)
    if use_step_clock:
        step_params = params.get("step_clock") or {}
        state = step_clock_state(env, step_params, update=params.get("update_step", False))
        if state is not None:
            phase = state.get("phase")
            if isinstance(phase, torch.Tensor):
                stance = 0.5 * (1.0 + torch.sin(phase))
                other = 1.0 - stance
                weights = torch.stack((stance, other), dim=-1)
                return weights[..., :num_legs]
    generator = getattr(env, "gait_generator", None)
    if generator is not None:
        weights = getattr(generator, "stance_weights", None)
        if isinstance(weights, torch.Tensor):
            return weights[..., :num_legs]
    phase = getattr(env, "gait_phase", None)
    if not isinstance(phase, torch.Tensor):
        phase = getattr(env, "_pm01_internal_phase", None)
        if not isinstance(phase, torch.Tensor):
            phase = torch.zeros(getattr(env, "num_envs", 1), device=_get_device(env))
        dt = getattr(env, "step_dt", None) or getattr(env, "dt", None) or 0.02
        speed = getattr(env, "command_manager", None)
        freq = 1.0
        if speed is not None:
            command = getattr(speed, "base_velocity", None)
            if isinstance(command, Mapping):
                maybe_freq = command.get("frequency")
                if isinstance(maybe_freq, torch.Tensor):
                    freq = maybe_freq.mean().item()
                elif isinstance(maybe_freq, (int, float)):
                    freq = float(maybe_freq)
        phase = (phase + dt * freq * 2.0 * torch.pi) % (2.0 * torch.pi)
        env._pm01_internal_phase = phase
    stance = 0.5 * (1.0 + torch.sin(phase))
    other = 1.0 - stance
    return torch.stack((stance, other), dim=-1)[..., :num_legs]


def swing_curve(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    weights = stance_curve(env, params)
    return 1.0 - weights


def _resolve_body_indices(data: Any, body_names: Iterable[str], device: torch.device) -> Optional[torch.Tensor]:
    if body_names is None:
        return None
    indices = []
    name_to_idx = getattr(data, "body_name_to_index", None)
    if isinstance(name_to_idx, Mapping):
        for name in body_names:
            idx = name_to_idx.get(name)
            if idx is not None:
                indices.append(idx)
    if indices:
        return torch.tensor(indices, device=device, dtype=torch.long)
    return None


def contact_mask(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    threshold = 1.0 if params is None else params.get("threshold", 1.0)
    num_feet = 2 if params is None else params.get("num_feet", 2)
    body_names = [] if params is None else params.get("body_names", [])

    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, num_feet)

    contact_forces = getattr(data, "net_contact_forces_w", None)
    if contact_forces is None:
        return _zeros(env, num_feet)

    indices_tensor = None
    if body_names:
        indices_tensor = _resolve_body_indices(data, body_names, contact_forces.device)
    if indices_tensor is None:
        total_bodies = contact_forces.shape[1]
        start = max(0, total_bodies - num_feet)
        end = min(total_bodies, start + num_feet)
        indices_tensor = torch.arange(start, end, device=contact_forces.device)

    if indices_tensor.numel() == 0:
        return _zeros(env, num_feet)

    limited_indices = indices_tensor[:num_feet]
    forces_selected = torch.index_select(contact_forces, 1, limited_indices)
    vertical = forces_selected[..., 2]
    mask = (vertical > threshold).to(contact_forces.dtype)
    return mask


def height_measurements(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    num_samples = 0 if params is None else params.get("num_samples", 0)

    terrain = getattr(env, "terrain", None)
    candidates: Iterable[torch.Tensor] = []
    if terrain is not None:
        for attr in ("height_measurements", "height_samples", "measured_heights"):
            candidate = getattr(terrain, attr, None)
            if isinstance(candidate, torch.Tensor):
                candidates = [candidate]
                break
    if hasattr(env, "height_scanner"):
        scanner = env.height_scanner
        for attr in ("heights", "height_measurements"):
            candidate = getattr(scanner, attr, None)
            if isinstance(candidate, torch.Tensor):
                candidates = [candidate]
                break
    for tensor in candidates:
        if tensor.shape[-1] >= num_samples:
            clamped = torch.clamp(tensor, -1.0, 1.0)
            return clamped[..., :num_samples]
    if num_samples <= 0:
        return _zeros(env, 0)
    return _zeros(env, num_samples)


def gait_phase_features(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    step_params = params.get("step_clock") if params is not None else None
    state = step_clock_state(env, step_params, update=params.get("update_step", False) if params else False)
    if state is None:
        return _zeros(env, 2)
    phase = state.get("phase")
    if not isinstance(phase, torch.Tensor):
        return _zeros(env, 2)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
