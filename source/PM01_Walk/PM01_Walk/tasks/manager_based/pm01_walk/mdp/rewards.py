# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Mapping

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.stack((quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]), dim=-1)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_vec = quat[..., 1:]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(quat), vec)


def _convert_to_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, (float, int)):
        return torch.tensor(float(value), dtype=torch.float32, device=device)
    if isinstance(value, (list, tuple)):
        try:
            return torch.tensor(value, dtype=torch.float32, device=device)
        except Exception:  # noqa: BLE001
            return None
    return None


def _extract_lin_vel_xy(candidate: Any, device: torch.device, num_envs: int) -> torch.Tensor | None:
    tensor = _convert_to_tensor(candidate, device)
    if tensor is not None:
        if tensor.ndim == 0:
            tensor = tensor.expand(2)
        if tensor.ndim == 1:
            if tensor.shape[0] >= 2:
                tensor = tensor[:2].unsqueeze(0)
            else:
                return None
        else:
            tensor = tensor[..., :2]
        if tensor.shape[0] == num_envs:
            return tensor.reshape(num_envs, 2)
        if tensor.shape[0] == 1:
            return tensor.expand(num_envs, 2)
        if num_envs == 1:
            return tensor.reshape(1, 2)
        return None

    if isinstance(candidate, Mapping):
        for key in ("command", "desired_command", "lin_vel", "lin_vel_xy", "linear_velocity", "value"):
            if key in candidate:
                result = _extract_lin_vel_xy(candidate[key], device, num_envs)
                if result is not None:
                    return result
        x_val = None
        y_val = None
        for key in ("lin_vel_x", "linear_velocity_x", "command_x", "target_lin_vel_x"):
            if key in candidate:
                x_val = candidate[key]
                break
        for key in ("lin_vel_y", "linear_velocity_y", "command_y", "target_lin_vel_y"):
            if key in candidate:
                y_val = candidate[key]
                break
        if x_val is not None and y_val is not None:
            x_tensor = _convert_to_tensor(x_val, device)
            y_tensor = _convert_to_tensor(y_val, device)
            if x_tensor is not None and y_tensor is not None:
                if x_tensor.ndim == 0:
                    x_tensor = x_tensor.expand(num_envs)
                if y_tensor.ndim == 0:
                    y_tensor = y_tensor.expand(num_envs)
                if x_tensor.ndim == 1 and y_tensor.ndim == 1:
                    if x_tensor.shape[0] == 1:
                        x_tensor = x_tensor.expand(num_envs)
                    if y_tensor.shape[0] == 1:
                        y_tensor = y_tensor.expand(num_envs)
                    if x_tensor.shape[0] == num_envs and y_tensor.shape[0] == num_envs:
                        return torch.stack((x_tensor, y_tensor), dim=-1)

    for key in ("command", "desired_command", "lin_vel", "lin_vel_xy", "linear_velocity", "value"):
        if hasattr(candidate, key):
            result = _extract_lin_vel_xy(getattr(candidate, key), device, num_envs)
            if result is not None:
                return result

    x_attr = None
    y_attr = None
    for key in ("lin_vel_x", "linear_velocity_x", "command_x", "target_lin_vel_x"):
        if hasattr(candidate, key):
            x_attr = getattr(candidate, key)
            break
    for key in ("lin_vel_y", "linear_velocity_y", "command_y", "target_lin_vel_y"):
        if hasattr(candidate, key):
            y_attr = getattr(candidate, key)
            break
    if x_attr is not None and y_attr is not None:
        x_tensor = _convert_to_tensor(x_attr, device)
        y_tensor = _convert_to_tensor(y_attr, device)
        if x_tensor is not None and y_tensor is not None:
            if x_tensor.ndim == 0:
                x_tensor = x_tensor.expand(num_envs)
            if y_tensor.ndim == 0:
                y_tensor = y_tensor.expand(num_envs)
            if x_tensor.ndim == 1 and y_tensor.ndim == 1:
                if x_tensor.shape[0] == 1:
                    x_tensor = x_tensor.expand(num_envs)
                if y_tensor.shape[0] == 1:
                    y_tensor = y_tensor.expand(num_envs)
                if x_tensor.shape[0] == num_envs and y_tensor.shape[0] == num_envs:
                    return torch.stack((x_tensor, y_tensor), dim=-1)

    return None


def _get_command_lin_vel_xy(
    env: ManagerBasedRLEnv, command_name: str, device: torch.device, num_envs: int
) -> torch.Tensor:
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return torch.zeros((num_envs, 2), device=device)

    candidates = []
    for attr in ("get_command", "get"):
        if hasattr(manager, attr):
            getter = getattr(manager, attr)
            try:
                candidate = getter(command_name)
            except TypeError:
                try:
                    candidate = getter(name=command_name)
                except Exception:  # noqa: BLE001
                    candidate = None
            except Exception:  # noqa: BLE001
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
                break

    if not candidates:
        for attr in ("commands", "data"):
            container = getattr(manager, attr, None)
            if isinstance(container, Mapping):
                candidate = container.get(command_name)
                if candidate is not None:
                    candidates.append(candidate)
                    break

    if not candidates:
        candidate = getattr(manager, command_name, None)
        if candidate is not None:
            candidates.append(candidate)

    for candidate in candidates:
        tensor = _extract_lin_vel_xy(candidate, device, num_envs)
        if tensor is not None:
            return tensor

    return torch.zeros((num_envs, 2), device=device)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def base_velocity_command_alignment(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """鼓励机器人沿着指令的平面速度方向前进。"""

    # 推断设备和环境数量
    num_envs = getattr(env, "num_envs", None)
    device = _as_device(getattr(env, "device", None))

    # 获取机器人线速度（机体坐标系）
    asset: Articulation = env.scene[asset_cfg.name]
    data = getattr(asset, "data", None)

    lin_vel_b = None
    if data is not None:
        candidate = getattr(data, "root_lin_vel_b", None)
        if isinstance(candidate, torch.Tensor):
            lin_vel_b = candidate.to(device=device)
            if num_envs is None:
                num_envs = lin_vel_b.shape[0]
        else:
            vel_w = getattr(data, "root_lin_vel_w", None)
            quat = getattr(data, "root_quat_w", None)
            if isinstance(vel_w, torch.Tensor) and isinstance(quat, torch.Tensor):
                lin_vel_b = _quat_rotate_inverse(quat.to(device=device), vel_w.to(device=device))
                if num_envs is None:
                    num_envs = lin_vel_b.shape[0]

    if num_envs is None:
        num_envs = 1

    if lin_vel_b is None:
        lin_vel_b = torch.zeros((num_envs, 3), device=device)
    elif lin_vel_b.ndim == 1:
        lin_vel_b = lin_vel_b.unsqueeze(0)
    if lin_vel_b.shape[0] != num_envs:
        if lin_vel_b.shape[0] == 1:
            lin_vel_b = lin_vel_b.expand(num_envs, -1)
        else:
            lin_vel_b = lin_vel_b.reshape(num_envs, -1)

    commanded_xy = _get_command_lin_vel_xy(env, command_name, device, num_envs)
    #print('commanded_xy:', commanded_xy)

     # 归一化指令速度方向

    commanded_norm = torch.linalg.norm(commanded_xy, dim=1, keepdim=True)
    commanded_dir = torch.where(commanded_norm > 1e-6, commanded_xy / commanded_norm, torch.zeros_like(commanded_xy))

    actual_xy = lin_vel_b[..., :2]
    #print('actual_xy before pad:', actual_xy)
    if actual_xy.shape[1] < 2:
        pad = torch.zeros((num_envs, 2 - actual_xy.shape[1]), device=device, dtype=actual_xy.dtype)
        actual_xy = torch.cat((actual_xy, pad), dim=1)

    progress = torch.sum(actual_xy * commanded_dir, dim=1)
    #print('progress:', progress)
    return progress
