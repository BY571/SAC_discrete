import torch
import numpy as np
from torch.distributions import Categorical
def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        
        a = np.random.uniform(low=0.0, high=1.0, size=env.action_space.n) 
        action_prob = (a - np.min(a))/np.ptp(a)
        dist = Categorical(torch.from_numpy(action_prob).float())
        action = dist.sample().numpy()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action_prob, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
