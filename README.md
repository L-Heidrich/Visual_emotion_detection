# Visual emotion detection using a CNN 


### TL;DR: A convolutional neural network trained recognize a persons mood based on their face expression.

## About <br>
The repository serves the use of me applying best practices in model training and deploying a model for inference. <br>
For that purpose it includes a test script which can be used to call the model via gcp where I hosted it using cloudrun. <br>
In test_app.py I give an example how the model can be called.  <br>
In the source code I wrote functions to retrieve and adjust the data for the model, train models with different hyper parameters and eventually validate them on a validation and test set.

## Links
The data comes from huggingface https://huggingface.co/datasets/FastJobs/Visual_Emotional_Analysis
