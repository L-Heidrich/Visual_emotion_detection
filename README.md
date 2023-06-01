# Visual_emotion_detection


### TL;DR: A convolutional neural network trained recognize a persons mood based on their face expression.

## About <br>
The reppository also includes a test script which can be used to call the model via gcp where I hosted it using cloudrun. <br>
It serves the use of me applying best practices in model training and deploying a model for inference. <br>
In test_app.py I give an example how the model can be called.  <br>
In the source code I wrote functions to retrieve and adjust the data for the model, train models with different hyper parameters and eventually validate them on a validation set.

## Links
The data comes from huggingface https://huggingface.co/datasets/FastJobs/Visual_Emotional_Analysis
