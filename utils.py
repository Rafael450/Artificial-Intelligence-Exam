

def reward_engineering_frogger(lives_count, forward_count, extra_action_rewards,is_past_road, has_started, state, action, reward, next_state, done, info):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 2).
    :param action: action.
    :type action: int.
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 2).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :return: modified reward for faster training.
    :rtype: float.
    """
    if reward > 0:
          has_started = True
    if has_started:
          # Additional rewards
          if action == 1:
              forward_count += 1
          if action == 4:
              forward_count -= 1
          reward += extra_action_rewards[action]
          if forward_count == 6 and not is_past_road:
              reward = 100
              is_past_road = True
          # Next Life
          if  info['lives'] < lives_count:
              is_past_road = False
              lives_count = info['lives']
              forward_count = 0
              reward = -100

    return reward


