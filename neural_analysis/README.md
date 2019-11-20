# Recipes to run neural analysis

These recipes assumes that the current working directory is the root of this repo.

This repo also assumes that you have access to the electrophysiology data from Freiwald & Tsao (2010) that we analyze in the paper. Even though we cannot share that data in this repo, you can get access by requesting it from the corresponding authors of the experimental study.

Example recipe to run the EIG network finetuned on the FIV faces.

```
cd neural_analysis
python test.py eig_classifier fiv --segment                    # run the finetuned EIG network on the FIV stimuli
python compare_neurons_models.py eig_classifier_fiv.hdf5 fiv   # compare neural data and model predictions.
```

To test on the FIV-S imageset (synthetic counterpart of the FIV images), you can use the following to run the network without the initial segmentation step.
```
python test.py eig_classifier bfm                              # run the EIG network on the FIV-S stimuli without the initial segmentation step
```

And to run with the initial segmentation step.
```
python test.py eig_classifier bfm --segment                    # run the EIG network on the FIV-S stimuli with the initial segmentation step
```

You can generate RSA matrices, do linear-decomposition analysis, and perform quantitative comparisons to neural data using

```
python compare_neurons_models.py eig_classifier_bfm.hdf5 bfm
```
