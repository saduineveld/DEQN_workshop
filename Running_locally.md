# Running DEQN_workshop code locally
- you need to run this code with specific (python) packages
- these are all set in Nuvolos, which also stores the output locally
- to run the code locally you need to use software to manage the environment (packages & dependencies). Example software: conda, venv
- you should be able to create the appropriate environment from the "environment.yaml" (folder: Day_2/DEQN_library)
- BUT this didn't work on my desktop
- I created a base environment (base_env.yaml) using conda, which includes sufficient packages to run the following models: (XX to be filled in XX)
- below I describe how to create this environment yourself using conda
- installing the same package with pip using venv should also work


## Creating environment "env_tf212_p311" from DEQN.yaml using conda
conda env create -f env_tf212_p311.yaml
conda activate env_tf212_p311

## Creating environment similar to env_tf212_p311 from scath using conda
- packages to install are (in brackets the versions I installed):
python (3.11.9)
tensorflow (2.12)
XX hydra, yaml, packaging? add conda-forge channel? XX

Note: it's best to first install packages that can be installed with conda, and then install the packages that can only be installed through pip
- conda create --<environment_name> (create environment)
- conda activate <environment_name>
- conda install python==3.11.9

- pip install tensorflow==2.12