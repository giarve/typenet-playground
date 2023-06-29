This is just a quick prototype to learn the (deep learning) concepts behind keystroke biometric systems and it is not meant to reach the same performance as TypeNet.

# Instructions

First run the notebook `extract_transform_dataset.ipynb` to extract the dataset and transform it into a faster-to-load format than csv (feather). This will improve training time as we do multiple passes (number of epochs) over the dataset.

Then run `python typenet_training.py` to start training the model on the transformed dataset.

You can clone the notebook `typenet_evaluation.ipynb` to pick the best validation model (and thresholds) in case you train it multiple times or save multiple checkpoints. Bear in mind that you will have to increase the validation set size and change the variable names to follow the validation set instead of the test set.

When training the model make sure to pick a smaller learning rate if loss gets stuck in the margin value (1.5 by default). You may need to restart training the triplet loss network a couple of times, unless you added triplet mining.

# References

A. Acien, A. Morales, J.V. Monaco, R. Vera-Rodriguez, J. Fierrez, "TypeNet: Deep Learning Keystroke Biometrics," in IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM), vol. 4, pp. 57 - 70, 2022. 

https://arxiv.org/abs/2101.05570
https://github.com/BiDAlab/TypeNet

---------------------------

V. Dhakal, A. M. Feit, P. O. Kristensson, and A. Oulasvirta. Observations on typing from 136 million keystrokes, in Proc. of the ACM CHI Conference on Human Factors in Computing Systems, 2018.

https://userinterfaces.aalto.fi/136Mkeystrokes/
