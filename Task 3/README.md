In the Project 3 our task was to take a pretrained model, and train and fine tune to model in such way that the unseen triplets (A, B, C) can be predicted. The Project was solved by following the template provided and completing the incomplete code. In `generate_embeddings` the task was to preprocess the images in the dataset, by using a pretrained model. The pretrained model used in this case is resnet50 as suggested. Late in the `get_data` function the triplets were loaded to extract the features and labels for training classifer. The class `Net` was implemented with the help of my teammates. This class is the architecture of the classifier.

Finally in the function `train_model` the model was trained and the loss was calculated. The functions is taking train_loader as an input from which the features and the labels are extracted. Then for a certain amount of epoches (20 at the end) the model is trained and the training loss is calculated as the average loss across all batches. 

Some improvements for later should have been the use of GPU instead of CPU which made the training process very slow.