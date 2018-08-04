# bsy-dqn-atari

A deep-q network (DQN) for the OpenAI Gym Atari domain.  bsy-dqn-atari learns to play Atari games from pixels at or above human levels.

[![Breakout 864 Score](https://github.com/benbotto/bsy-dqn-atari/raw/readme/asset/img/BreakoutNoFrameskip-v4__2018_07_01__08_10.max__Episode_41__Score_864.gif)](https://youtu.be/CP3nZMa3cis)

# About

bsy-dqn-atari combines the algorithms published in three reinforcement learning papers: [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/), [
Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), and [
Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).  The software is written in python and uses Keras with the Tensorflow-gpu 1.8 backend.

# Implementation Differences

The three papers referenced above train on batches of 32 samples; bsy-atari-dqn trains on batches of 64 samples.  The speed, capacity, and bandwith of graphics cards has increased significantly in the past few years.  Increasing the batch size resulted in a score increase without a noticible effect on performance when using an NVIDIA 1080 GTX.

This implementation uses [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) to clip error gradients rather than clipping the error term.  In the original Nature DQN paper, the authors note that "We also found it helpful to clip the error term from the update [...] to be between -1 and 1."  There are two ways that this can be interpreted: ensure that the difference between the actual and expected reward is in the range [-1, 1]; clip the error _gradient_ such that it falls between [-1, 1].  The correct interpretation has been contested by the ML community ([1](https://blog.openai.com/openai-baselines-dqn/), [2](https://www.reddit.com/r/MachineLearning/comments/4dnyiz/question_about_loss_clipping_on_deepminds_dqn/), [3](https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn)), but [DeepMind's code](https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn) shows that they used the former and clipped the error term.  bsy-dqn-atari, however, clips the gradient using Huber Loss, which empirically seems to work better, makes sense mathematically, and seems to be the agreed-upon community standard.

Importance sampling weights are multipled by the error ___before__ applying [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)_.  This differs from the native Tensorflow [huber_loss](https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/ops/losses/losses_impl.py#L375) implementation, and from [Open AI's DQN](https://github.com/openai/baselines/blob/master/baselines/deepq/build_graph.py#L413) implementation.  The change resulted in a large increase in score and makes sense mathematically.  With this implementation, errors exceeding 1, when scaled down by the IS weights, may fall within the squared loss region, thereby resulting in a smaller weight adjustment.  In Tensorflow's and Open AI's implementations, errors exceeding 1 will always fall into the linear loss region (the derivitave will always be 1).  Note that the three referenced papers do not use Huber Loss, but rather use mean squared error with error clipping.

The scoreboard is not clipped out of image samples.  In [the original DQN](https://arxiv.org/abs/1312.5602) implementation images are preprocessed as follows:

  1) Samples are taken every four frames.  This is called frame skipping, and increases the training speed.
  2) The maximum pixel values of the current image and the previous are taken to prevent effects caused by flickering (i.e. sprites that only show up every other frame).
  3) A grayscale of the image is taken.  This reduces the size of the image by a factor of three because only one channel is needed as opposed to three, red, green, and blue.
  4) The image is scaled down by a factor of two (to 110x84), then an 84x84 square is clipped from that.
  5) The last four images are stacked together to produce a sample.  This way velocity can be determined.

bsy-atari-dqn differs in step 4.  The images are scaled down to 84x84 directly rather than clipping out a square.
