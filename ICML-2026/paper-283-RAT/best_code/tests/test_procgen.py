from procgen import ProcgenEnv
from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)


def test_procgen():
    venv = ProcgenEnv(num_envs=4, env_name='dodgeball', distribution_mode='easy')
    venv = VecExtractDictObs(venv, "rgb")

    venv.reset()
    for _ in range(2048):
        venv.step(venv.action_space.sample())

    venv = VecMonitor(venv=venv)
    venv = VecNormalize(venv=venv) # img transform and reward normalization