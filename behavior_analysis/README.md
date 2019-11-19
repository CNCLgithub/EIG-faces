# Recipes to run behavioral analysis

These recipes assumes that the current working directory is the root of this repo.

# Identity matching task

```
cd behavior_analysis/identity_matching
python test.py eig_classifier     # run the EIG-classifier network on the behavioral stimuli
python test.py vgg                # run the VGG network on the behavioral stimuli
python test.py vgg_raw            # run the VGG-raw network on the behavioral stimuli
python test_pixels.py             # simple image matching strategy using pixels.
#python test_sift.py              # simple image matching strategy using SIFT. This output is delivered with the repo because it requires propreitary software.
python compare_behavior_models.py

```

# Lighting direction judgment in hollow face illusion

Recipe to run the EIG network.

```
python test.py eig_classifier                  # run the EIG-classifier 
python compare_model_data_lighting_direction.py ./output/eig_classifier.hdf5   # behavioral comparison
```

# Face depth judgment in hollow face illusion

Recipe to run the EIG network.

```
python test.py eig_classifier
matlab                                        # start matlab
measure_face_depth                            # obtain predicted face depths
quit                                          # quit matlab
python compare_model_data_depth_judgment.py eig_classifier_predicted_depth.txt
```



