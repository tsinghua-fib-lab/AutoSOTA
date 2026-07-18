#!/bin/bash
# AMS-TD3 evaluation on Quadruped-Run
# Reproduces the paper's main result (Table 2)
set -e

# Start virtual display for headless rendering
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
XVFB_PID=$!
sleep 1

MUJOCO_GL=glfw PYOPENGL_PLATFORM=glfw DISPLAY=:99 python /repo/AMS_TD3.py   --env_id dm_control/quadruped-run-v0   --total_timesteps 500000   --num_envs 8   --seed 0   --K_neighbors 8   --neighborhood_radius 0.2   --eval_frequency 50000   --eval_episodes 10

kill $XVFB_PID 2>/dev/null || true
