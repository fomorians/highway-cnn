# Highway Networks

Starter project for training highway networks on Fomoro.

Check out the [blog post](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa).

## Training

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for Fomoro.
2. Clone the repo: `git clone https://github.com/fomorians/highway-cnn.git && cd highway-cnn`
3. Create a new model: `fomoro model create`
4. Start training: `fomoro session start`
5. Follow the logs: `fomoro session logs -f`

### Local Setup

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/highway-cnn.git && cd td-gammon`
3. Run training: `python main.py`
