import numpy as np
from colorama import Fore
import torch

@torch.no_grad()
def evaluate_agent(env, agent, logfile, step_train, episode_train, num_episodes, eval_idx, maximum_speed, best_score, steps_array, scores_eval, train):
    logfile.reset()
    scores_eval_idx = []
    steps = 0
    for ep in range(num_episodes):
        done = False
        if not train:
            env.set_task_idx(ep)
        observation = env.reset()
        agent.reset(obs=observation)
        score = 0
        while not done:
            action, controls = agent.choose_action(
                obs=observation, training=False, step=step_train)

            observation_, reward, done, info = env.step(controls)
            score += reward

            logfile.add_data(episode=ep, step=steps, info=info,
                             observation=observation_, action={'speed': [float(action[0])*maximum_speed], 'steer': [float(action[1])]}, reward=reward, done=done)

            observation = observation_
            steps += 1

        scores_eval_idx.append(score)
        
        if logfile.is_full():
            if not train:
                episode_train = ep
                step_train = steps
            logfile.save_episodes(episode=episode_train, step=step_train,
                          scores=scores_eval, steps_array=steps_array, eval_idx=eval_idx)
        
         
        if not train:
            episode_train = ep
            step_train = steps
            print(
                f" {Fore.YELLOW} episode {episode_train}, step: {step_train}, score: {score:.1f} {Fore.RESET}")

        
    mean_score = np.mean(scores_eval_idx)
    scores_eval.append(mean_score)
    steps_array.append(step_train)

    if train and (mean_score > best_score):
        best_score = mean_score
        agent.save_models()
        logfile.update_episodes_saved(episode=episode_train, step=step_train)

    print(
        f" {Fore.YELLOW} evaluation: {eval_idx}, episode {episode_train}, step: {step_train}, average score: {mean_score:.1f} {Fore.RESET}")

    if not train:
        episode_train = ep
        step_train = steps
        
    logfile.save_episodes(episode=episode_train, step=step_train,
                          scores=scores_eval, steps_array=steps_array, eval_idx=eval_idx)

    eval_idx += 1

    return eval_idx, best_score, scores_eval, steps_array
