# EIG-faces
Efficiently inverting a probabilistic graphics program with an inference network. Includes models, neural analyses, and behavioral experiments. 

# Setting up your environment
(1) You can use `conda` for seting up your environment. First create a new environment with `python=3.6`, activate your new environment, and do the following installations.

```
conda install -c conda-forge matplotlib            # install matplotlib, the plotting library needed to reproduce figures in the paper
conda install -c anaconda scipy                    # install scipy
conda install pytorch torchvision -c pytorch       # install pytorch and torchvision
conda install -c anaconda configparser             # install configparser
conda install -c anaconda PIL                      # install PIL image processing library
conda install h5py                                 # install h5py to process datasets
conda install -c anaconda pandas                   # install pandas
conda install -c anaconda scikit-learn             # install scikit-learn
pip install opencv-contrib-python-headless         # install opencv using pip.
```

(2) Update your PYTHONPATH environment variable to include the root of the project.
```
export PYTHONPATH=${PYTHONPATH}:$ROOT
```
where `$ROOT` is an environment variable pointing to the root of this repo.

(3) Download pretrained weights and our distributed weights. At the root run:
```
chmod +x download_network_weights.sh
./download_network_weights.sh
```

(4) Adjust `default.conf` to your needs by copying it to `user.conf` and editing its contents. What matters is its second half, `[PATHS]`.
