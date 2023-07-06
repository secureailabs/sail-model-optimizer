# sail-model-optimizer
a general feature selection and hyper parameter optimization library.

This library is intended to be used as a method to automate a great deal of the modle optimization and feature selection pipeline. There are 3 main components here.

1. <b>Data pre-processing</b> represents the step to be done onthe SAIL UI. This includes cohort selsection and removing any columns which are irrelevant or will otherwise interfere with the training.
2. <b> Model Training </b> is where each individual model type is trainied with different feature and hyperparameter permutations. Each model run is saved to cache for future reference. The most performant models are saved in th results.
3. <b> Model Validation </b> Once optimal models are selected, they are validated individually and then combined into a metaclassifier. Perfromance statistics are visualised and saved with the results.

In order to see a demo of this, try running the scripts contained in the flf directory.

In order to expose this functionality to the UI, we host a fastAPI server to run this library. This fastAPI server is hosted on a docker container which we spin up when the request is made from the frontend. To deploy this server locally run the following:



