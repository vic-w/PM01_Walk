# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""自定义PM01行走任务的MDP函数扩展。"""

from __future__ import annotations

from typing import Any, Optional

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as _base_velocity_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401,F403


_PRINT_INTERVAL = 200


def _resolve_sensor(env: Any, sensor_cfg: Any) -> Optional[Any]:
    """根据传感器配置找到具体的传感器实例。"""
    if sensor_cfg is None:
        return None

    sensor_name = getattr(sensor_cfg, "name", None)
    scene = getattr(env, "scene", None)

    if scene is None or sensor_name is None:
        return None

    sensors = getattr(scene, "sensors", None)
    if isinstance(sensors, dict) and sensor_name in sensors:
        return sensors[sensor_name]

    get_entity = getattr(scene, "get_entity", None)
    if callable(get_entity):
        try:
            return get_entity(sensor_name)
        except Exception:
            return None

    return None


def _extract_contact_forces(sensor: Any) -> Optional[torch.Tensor]:
    """尝试从传感器对象中提取接触力张量。"""
    if sensor is None:
        return None

    data = getattr(sensor, "data", None)
    if data is None:
        return None

    for attr in ("net_forces_w", "net_forces", "forces_w", "forces"):
        forces = getattr(data, attr, None)
        if isinstance(forces, torch.Tensor):
            return forces
    return None


def _summarize_forces(forces: torch.Tensor) -> str:
    """为调试输出创建接触力的概要信息。"""
    if not isinstance(forces, torch.Tensor) or forces.numel() == 0:
        return "无有效接触力数据"

    forces_cpu = forces.detach().to("cpu")
    abs_forces = forces_cpu.abs()
    max_force = abs_forces.max().item()
    mean_force = abs_forces.mean().item()
    sample = forces_cpu[0].tolist() if forces_cpu.ndim > 0 else [forces_cpu.item()]
    return (
        f"max={max_force:.3f}N, mean={mean_force:.3f}N, "
        f"first_sample={', '.join(f'{value:.3f}' for value in sample[:min(len(sample), 3)])}"
    )


def feet_air_time_positive_biped(*args: Any, **kwargs: Any):
    """包装原始的脚部腾空时间奖励，并打印接触力调试信息。"""
    sensor_cfg = kwargs.get("sensor_cfg")
    if sensor_cfg is None and len(args) >= 3:
        sensor_cfg = args[2]

    env = args[0] if args else None
    sensor = _resolve_sensor(env, sensor_cfg)
    forces = _extract_contact_forces(sensor)

    call_count = getattr(feet_air_time_positive_biped, "_call_count", 0)
    if call_count % _PRINT_INTERVAL == 0:
        if forces is not None:
            summary = _summarize_forces(forces)
            print(f"[FeetAirTimeDebug] 第{call_count}次调用：接触力摘要 -> {summary}")
        else:
            sensor_name = getattr(sensor_cfg, "name", "未知传感器") if sensor_cfg else "未配置"
            print(
                f"[FeetAirTimeDebug] 第{call_count}次调用：未能获取传感器\"{sensor_name}\"的接触力数据"
            )

    result = _base_velocity_mdp.feet_air_time_positive_biped(*args, **kwargs)

    if call_count % _PRINT_INTERVAL == 0:
        if isinstance(result, torch.Tensor) and result.numel() > 0:
            reward_preview = result.detach().to("cpu")[0].item()
            print(f"[FeetAirTimeDebug] 第{call_count}次调用：奖励样本值 -> {reward_preview:.6f}")
        else:
            print(f"[FeetAirTimeDebug] 第{call_count}次调用：奖励值类型 -> {type(result)}")

    feet_air_time_positive_biped._call_count = call_count + 1
    return result


__all__ = [name for name in globals().keys() if not name.startswith("_")]
