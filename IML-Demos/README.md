# Intro to ML Notebooks

## Environment (Optional) 
Create a conda environment where everything should run. 
```bash
$ conda create -n intro-ml python=3.9
$ conda activate intro-ml
```

## Install requirements
To install the requirements
```bash
$ pip install -r requirements.txt
```
In order to install Tensorflow, please run
```
$ pip install tensorflow
```
If you have a Mac with a M1 chip, please instead call
```
$ pip install tensorflow-macos
```
Lastly, for some notebooks, we need [JAX](https://jax.readthedocs.io/en/latest/). For installation on CPU, please run
```
$ pip install --upgrade "jax[cpu]"
```
The installation might vary depending on which operating system you are using. Please consult JAX's [installation guide](https://github.com/google/jax#installation) for more information.

## Run a jupyter notebook server
If you want jupyter notebook support, run the following command (inside the `intro-ml` environment)
```
$ python -m ipykernel install --user --name=intro-ml
```
Start a jupyter notebook server by running 
```bash
$ jupyter notebook 
```
We are using iPython widgets for visualizations. For this, please run
```
$ jupyter nbextension enable --py widgetsnbextension
```
Do not forget to select the `intro-ml` kernel!


## Run a jupyter notebook in a remote cluster
Go to Server using

```bash
$ ssh username@ip_address 
```

On remote terminal run:
```bash
$ jupyter notebook --no-browser --port=7800 
```

On your local terminal run [explained](https://explainshell.com/explain?cmd=ssh+-N+-f+-L+localhost%3A8001%3Alocalhost%3A7800+username%40ip_address):
```bash
$ ssh -N -f -L localhost:8001:localhost:7800 username@ip_address 
```

Open web browser on local machine and go to http://localhost:8001/
