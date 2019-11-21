# Singularity Recipe

This is providing a singularity recipe file to build a singularity container to run EIG.

Using singularity replaces the installation of the conda environment.

(1) Install singularity
(2) Build container
```
sudo singularity build  sing.img singularity.recipe
```
(3) Run container (gives you an interactive shell within the singularity container). The EIG-faces data and code is stored outside of the container at /local/EIG-faces on the host machine
```
singularity shell -B /local/EIG-faces:/EIG-faces --nv sing.img
```
(4) Activate conda environment
```
. activate env  
```
(5) Proceed with step 3 of the original setup
