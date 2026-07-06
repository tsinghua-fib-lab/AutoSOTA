import gymnasium as gym
# 등록된 DMControl 환경 전체 ID 리스트 불러오기
from shimmy.registration import DM_CONTROL_SUITE_ENVS

dmlist = [f"dm_control/{'-'.join(item)}-v0" for item in DM_CONTROL_SUITE_ENVS]

for i in dmlist:
    print(i)