# AI-lligator

A CNN wrote in Python able to recognise between Crocodiles and Alligators.



## Quick guide

This code requires:
- Conda: https://www.anaconda.com/download/
- Python v. 3.9.18 (Tensorflow does not support newer version atm): https://www.python.org/downloads/release/python-3918/

First of all, install Conda from the link above and open a conda terminal.

Then create a venv with ```conda create -n <venv_name> python=3.9.18 anaconda```, this will create a conda venv with the python version you need to train your model.

Activate the venv with ```conda activate <venv_name>```.

At this point you can go to your clone of this repository and install Tensorflow and the other dependences.

Do this with ```pip install -r requirements.txt```.

Before you try to train your own model be sure that your GPU can work with Tensorflow, otherwise you can use your CPU or a TPU as well.



## Dataset

In this repo you won't find the dataset we used to train our model.

Below you can find a decent one to start with.

https://www.kaggle.com/datasets/rrrohit/crocodile-gharial-classification-fastai


Remember that you have to create the folder "dataset" and in there the following subdirectories:
    - Training
    - Validation
    - Testing



## Training

Once you've got the env and the dataset you can train your model, to do that simply type the following command in your conda prompt:

```python TrainingArc.py```



## Testing

To see how your model reacts to specific images you can use "Predition.py" by typing (always in your conda venv):

```python Prediction.py <image_path>```


This will give you the probability of the image being a crocodile one (if it's nearby 0 it means it's an Alligator, nearby 1 means it should be a Crocodile).



### Authors

Retr0mus: https://github.com/Retr0mus
Singh-0: https://github.com/Singh-0
senshi-desu: https://github.com/senshi-desu
