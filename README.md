# EIG-faces
Efficiently inverting a probabilistic graphics program with an inference network. Includes models, neural analyses, and behavioral experiments. 

# Setting up the BFM'09 model (generative model)

You need to obtain the Basel Face Model 2009's data in order to be able to run the matlab scripts and render new faces. The generative model is controlled via matlab scripts. This repo includes (adapted) versions of BFM09's core matlab scripts under `bfm09-generator/bfm_utils/PublicMM1/matlab`. However, we cannot share the model weights. You need to download them from the official website. 

(1) Go to https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details where you can find a link to the download the BFM09 model at the bottom.

(2) Request and download the model.

(3) Finally copy `01_MorphableModel.mat` to this repo, under `EIG-faces/bfm09-generator/bfm_utils/PublicMM1/`.

# Setting up your environment
(1) You can use `conda` for seting up your environment. We recommend setting up a fresh environment at the root of this project. First create a new environment with `python=3.6`, activate your new environment, and install a number of packages as shown below.

```
cd EIG-faces                                       # cd into the root of this directory
conda create -n env python=3.6                     # create a fresh environment
conda activate env                                 # activate your environment
conda install -c conda-forge matplotlib            # install matplotlib
conda install -c anaconda scipy                    # install scipy
conda install pytorch torchvision -c pytorch       # install pytorch and torchvision
conda install -c anaconda configparser             # install configparser
conda install h5py                                 # install h5py to process datasets
conda install -c anaconda pandas                   # install pandas
conda install -c anaconda scikit-learn             # install scikit-learn
```

(2) Update your PYTHONPATH environment variable to include the root of the project.
```
export PYTHONPATH=${PYTHONPATH}:$ROOT
```
where `$ROOT` is the full path to the root of this repo.

(3) Download pretrained weights and our distributed weights. At the root run:
```
chmod +x download_network_weights.sh
./download_network_weights.sh
```

(4) Adjust `default.conf` to your needs by copying it to `user.conf` and editing its contents. It contains paths to checkpoints, stimuli, etc. under `[PATHS]`.

# Infer and render using EIG

Here is a recipe to run the EIG model on a folder with input image files. Assuming you are at the root of the project (`EIG-faces`) and have your conda environment activated:

(1) 
```
cd infer_render_using_eig
python infer.py --imagefolder ./demo_images --segment
```

NOTE: Don't need to use `--segment` if the input images have clean background. So in that case, you would say
```
cd infer_render_using_eig
python infer.py --imagefolder ./my_demo_images  # where the folder ./my_demo_images contain images with clean backgrounds.
```

(2)
Now this will output an `.hdf5` file under `./output`. To render them, you need to call the matlab script in matlab. You may need to edit the `render.m` file to point it to that outputed `.hdf5` file.

```
matlab   # start a matlab session
render   # render
```

NOTE on MATLAB: Matlab scripts in this repo are tested in version 2016b and before. If you are using a newer version of MATLAB, you may need to replace the lines where the function `hardcopy` is called with the following. (`hardcopy` is discountined apparently.)

`img = print('-RGBImage');`



