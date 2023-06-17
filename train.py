import pickle

import numpy as np
import torch

from agent import DQN
from env import ImageEnv, SkillCheck


def evaluate(agent, n_evals=5):
    eval_env = SkillCheck()
    eval_env = ImageEnv(eval_env, skip_frames=1, stack_frames=1, initial_no_op=0)

    scores = 0
    for i in range(n_evals):
        (s, _), done, ret = eval_env.reset(), False, 0
        while not done:
            a = agent.act(s, training=False)
            s_prime, r, terminated, truncated, info = eval_env.step(a)
            s = s_prime
            ret += r
            done = terminated or truncated
        scores += ret
    return np.round(scores / n_evals, 4)


def main():
    env = SkillCheck()
    env = ImageEnv(env, skip_frames=1, stack_frames=1, initial_no_op=0)

    # Hyperparameters
    max_steps = int(1e6)
    eval_interval = 10000
    state_dim = (1, 84, 84)
    action_dim = env.action_space.n
    epsilon = 0.1
    epsilon_min = 0.005

    agent = DQN(state_dim, action_dim, epsilon=epsilon, epsilon_min=epsilon_min)

    (s, _) = env.reset()
    history = {'Step': [], 'AvgReturn': []}
    while True:
        a = agent.act(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.process((s, a, r, s_prime, terminated))

        s = s_prime
        if terminated or truncated:
            s, _ = env.reset()

        if agent.total_steps % eval_interval == 0:
            ret = evaluate(agent)
            print(f"Step: {agent.total_steps}, AvgReturn: {ret:.4f}")
            history['Step'].append(agent.total_steps)
            history['AvgReturn'].append(ret)

        if agent.total_steps > max_steps:
            break

    torch.save(agent.network.state_dict(), './results/dqn.pt')
    with open('./results/history.pkl', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    main()
