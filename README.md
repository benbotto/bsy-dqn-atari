# bsy-dqn-atari

A deep-q network (DQN) for the OpenAI Gym Atari domain.  bsy-dqn-atari learns to play Atari games from pixels at or above human levels.

[![Breakout 864 Score](https://github.com/benbotto/bsy-dqn-atari/raw/readme/asset/img/BreakoutNoFrameskip-v4__2018_07_01__08_10.max__Episode_41__Score_864.gif)](https://youtu.be/CP3nZMa3cis)

# About

bsy-dqn-atari combines the algorithms published in three reinforcement learning papers: [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/), [
Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), and [
Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).  The software is written in python and uses Keras with the Tensorflow-gpu 1.8 backend.

# Implementation Differences

The three papers referenced above train on batches of 32 samples; bsy-atari-dqn trains on batches of 64 samples.  The speed, capacity, and bandwith of graphics cards has increased significantly in the past few years.  Increasing the batch size resulted in a score increase without a noticible effect on performance when using an NVIDIA 1080 GTX.

This implementation uses [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) to clip error gradients rather than clipping the error term.  In the original Nature DQN paper, the authors note that "We also found it helpful to clip the error term from the update [...] to be between -1 and 1."  There are two ways that this can be interpreted: ensure that the difference between the actual and expected reward is in the range [-1, 1]; clip the error _gradient_ such that it falls between [-1, 1].  The correct interpretation has been contested by the ML community [1](https://blog.openai.com/openai-baselines-dqn/) [2](https://www.reddit.com/r/MachineLearning/comments/4dnyiz/question_about_loss_clipping_on_deepminds_dqn/) [3](https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn).  [DeepMind's code](https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn) shows that they used the former and clipped the error term.  bsy-dqn-atari, however, clips the gradient using Huber Loss, which empirically seems to work better and makes sense mathematically.
