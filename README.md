Completed as part of an Imperial College Coursework in Reinforcement Learning (CO424).

## Base Files:

- ```Coursework_Part_1.pdf``` containing all the main coursework instructions and questions.
- ```Tutorial.pdf``` which explains how to implement Deep Q-Learning through several stages. It is aligned with ```Coursework_Part_1.pdf```.
- ```starter_code.py``` providing Python 3 code which you will build upon during this tutorial and the associated coursework.
- ```environment.py``` in which the environment is implemented. **This file should not be modified**.
- ```torch_example.py``` which gives an example of a supervised learning experiment in PyTorch (see section 2 in ```Tutorial.pdf``` for more information).

## Requirements

pip install -r requirements.txt
```

This will install the following libraries (and their dependencies):

- ```torch``` 
- ```opencv-python```
- ```numpy```
- ```matplotlib```

## How to run a script ?

```shell script
python torch_example.py  # To launch the pytorch example script
python starter_code.py  # To launch the coursework script
```

## Techniques used:

Double Deep Q-Learning

Custom Experience Replay Buffer to sample uniformly across map areas

Free exploration before training

Continuous action space using Cross-Entropy Method
