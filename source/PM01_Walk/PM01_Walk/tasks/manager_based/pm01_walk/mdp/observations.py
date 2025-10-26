"""自定义观测项的实现。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch


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
    # X 轴
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # Y 轴
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1.0, torch.sign(sinp) * (torch.pi / 2.0), torch.asin(sinp))
    # Z 轴
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


def _difference_with_reference(current: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
    if reference is None:
        return current * 0.0
    return current - reference


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
        data = manager.data.get(action_name or "joint_pos")
        if isinstance(data, torch.Tensor):
            return data
    return _zeros(env, 0)


def dof_pos_reference_diff(env: Any, params: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
    del params
    data = _get_robot_data(env)
    if data is None:
        return _zeros(env, 0)

    current = getattr(data, "joint_pos", None)
    reference = None
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
        data = env.event_manager.data
        if isinstance(data, Mapping):
            candidate = data.get("rand_push_force")
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
        data = env.event_manager.data
        if isinstance(data, Mapping):
            candidate = data.get("rand_push_torque")
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
