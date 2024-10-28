# Running DEQN_workshop code locally
- you need to run this code with specific (python) packages
- these are all set in Nuvolos, which also stores the output locally
- to run the code locally you need to use software to manage the environment (packages & dependencies). Example software: conda, venv
- you should be able to create the appropriate environment from the "environment.yaml" (folder: Day_2/DEQN_library)
- BUT this didn't work on my desktop
- I created a base environment (DEQN_fork.yaml) using conda. This has been tested for the models: bm1972 
- below I describe how to create this environment yourself using conda
- you can set the folder where hydra puts the output (I adjusted this to a folder outside the repository)

## Creating environment DEQN_fork from yaml using conda
conda env create -f DEQN_fork.yaml
conda activate DEQN_fork

## Creating environment from scrath using conda
- use "conda install <packagename>" (and not "pip install") to install packages, because conda tracks packages to ensure compatibility
- add the conda-forge channel (conda-forge channel in general has more version options than the default channel)
- packages to install are (in brackets the latest compatible versions, as of 28.10.24, conda-forge supports for windows at the moment):
python (3.10)
tensorflow (2.10)
hydra-core (1.3.2)

- conda create --<environment_name> # create environment
- conda activate <environment_name> # activate environment
- conda install python==3.10
- conda install tensorflow==2.10
- conda install hydra-core