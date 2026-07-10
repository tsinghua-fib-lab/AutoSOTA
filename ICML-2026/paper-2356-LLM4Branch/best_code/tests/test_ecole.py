import ecole

def test_generator():
    generator = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=200, density=0.1)
    assert next(generator) != None

def test_branch():
    env = ecole.environment.Configuring(information_function = ecole.reward.NNodes())
    env.seed(42)
    generator = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=200, density=0.1)
    observation, action_set, reward_offset, done, info = env.reset(next(generator))
    _, _, _, _, info = env.step({})
    assert info >= 1.0