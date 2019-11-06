# Coursework (Part 1) - CO424 - Reinforcement Learning Part II

## Coursework Documents

You may find in this repository all the files that are necessary for completing the coursework:

- ```Coursework_Part_1.pdf``` containing all the main coursework instructions and questions.
- ```Tutorial.pdf``` which explains how to implement Deep Q-Learning through several stages. It is aligned with ```Coursework_Part_1.pdf```.
- ```starter_code.py``` providing Python 3 code which you will build upon during this tutorial and the associated coursework.
- ```environment.py``` in which the environment is implemented. **This file should not be modified**.
- ```torch_example.py``` which gives an example of DQN implementation using PyTorch (see section 2 in ```Tutorial.pdf``` for more information).

## Requirements

You need to use Python 3.6 or greater.

## Installing the environment on a Unix system (including the Lab computers)

We created this repository to ensure that everybody uses exactly the same versions of the libraries.

To install the libraries, start by cloning this repository and enter the created folder:

```shell script
git clone https://gitlab.doc.ic.ac.uk/lg4615/co424-reinforcement-learning-part-2-coursework.git
cd co424-reinforcement-learning-part-2-coursework
```

You cannot install the libraries directly on the Lab computers (since you do not have ```sudo``` rights)
Also, it is a good practice to have a specific virtual environment 
dedicated to the project.


That is why we advise you to generate a virtual environment (called ```venv``` here):

```shell script
python3 -m venv ./venv 
```

Then enter the environment:
```shell script
source venv/bin/activate
```

And install the libraries in the environment by launching the following command:
```shell script
pip install -r requirements.txt
```

This will install the following libraries in the virtual environment ```venv```:

- ```torch``` 
- ```opencv-python```
- ```numpy```

## How to run a script ?

Before launching your experiment, be sure to use the right virtual environment in your shell:
```shell script
source venv/bin/activate  # To launch in the project directory
```

Once you are in the right virtual environment, you can directly launch the scripts 
by using one of the following command:
```shell script
python torch_example.py  # To launch the pytorch example script
python starter_code.py  # To launch the coursework script
```

It is also possible to use the virtual environment tools already included in IDEs (such as PyCharm).

## Leaving the virtual environment

If you want to leave the virtual environment, you just need to enter the following command:
```shell script
deactivate
```