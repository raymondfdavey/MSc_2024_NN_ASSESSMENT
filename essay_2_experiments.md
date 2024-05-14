# experiment 1 LERNING RATES

* Experiments - Describe how you conducted each experiment, including any changes made to the baseline model that has already been described in the other Methodology sections. Explain the methods used for training the model and for assessing its performance on validation/test data.

## 1.1 LR

In order to investigate the the effect of learning rate on my BaselineNet (see above) models performance initial exploratory trails learning rates of 0.2 to 0.1e^-5 decreasing by a power of ten revealed the extremes of the effect learning rate had (no learning at lr - 0.0001), unstable learning at beahviour of the model. these invrestigation suggested a sensible range from which to select the 5 learning rates to test which were selected as 0.1, 0.075. 0,05, 0.025, 0.01]


Hyperparamaters and model architecture were kept constant across LR trials, as was the data used (including the train/test/val split) which was also seeded to preserve consistence across all trials.

For each learning rate, 5 trials were conducted. That is, 5 different models were instantiated, trained and evlauated for each learning rate. 

 For each of the 5 models trained for each learning rate, each model was initialised with a different one of 5 random seeds that were kept constant across learning test rate to ensure fair comparison. Each modelweas trained using mini-batch schoastic gradient descent as mentioned in the desciptions above. 

During training each batch was scored in terms of loss and accuracy, and these batch scores were averaged across the epoch to give the training loss and accuracy for that epoch. After training for each epoch, the model was was then token out of training mode - halting any gradient computations - and the validation set was iterated through in batches, with the batch losses and accuracies again being averaged to give a validation loss and accuracy for the epoch.

These epoch level performance metrics were stored for each of the 5 models for each learning rate for evaluations. Accumulating these measures across epochs rather than batches is a somewhat aribitraty although conventional approach. It is a convenient way to keep track of how many times the model has been exposed to all of the training data and is easy to understand when plotting performance graphs. 

Metrics for each run of each learning rate were plotted along with the average to visualise any variance in performance, and then the average performance of each LR were plotted together with validation performance smoothed with a window 5 to make the results clearer.  

Results on validation and testing perfomance were put into a table and the best perfroming LR was identified


By running this experiment with different learning rates and multiple random seeds, a comprehensive understanding of how the learning rate affects the model's performance can be obtained. The averaged results provide insights into the model's training progress, generalization ability, and final performance on the test set for each learning rate. The smoothed plots help visualize the trends and compare the performance across different learning rates.
The experiment's results can guide the selection of an appropriate learning rate for the BaselineNet model on this specific classification task, helping to identify the learning rate that achieves the best balance between training performance, generalization, and final test accuracy.

# 1.2 LR Scheduler

Having established above the performance of differnt learning rates it was clear that the model could tolerate a relatively high initial learning rate but this needed to drop significantly and arrive at approximately .. by the end of the epochs. 

A number of approached to learning rate scheduling are available, but it made sense to try a number of simple appraoches first, so a simple step decay and smooth decay function were plotted to see how LR would be affected and can be seen here. A lernig rate decay was chosen that would take the LR from x to y.

A smoother learning rate decay seemed to make more sense and worked well in trials, although it should be noted the stepped learning rate performed similarly.

The adjust_initial_learning_rate function is defined to adjust the learning rate during training. It takes the optimizer, current epoch, initial learning rate, and decay rate as input and returns the updated optimizer with the new learning rate. The learning rate is adjusted according to the formula: new_lr = initial_lr / (1 + decay_rate * epoch).

The performance of this lr rate scheduler was again tested by intitialising and training 5 models on the CIFAR-10 data as described above, with all other hyper params kept steady and initial learning rate selected as x with a desired end learning rate of y.

The schedulers performance across the 5 runs was averaged and compared to the performance of the most successful performing model without learning rate sheduling from the previous experiment 1.1 and results plotted. 


By running this experiment with learning rate decay, the impact of gradually reducing the learning rate on the model's performance can be observed. The averaged results and smoothed plots provide insights into how the model's training progress, generalization ability, and final performance on the test set are affected by learning rate decay.
The experiment's results can help determine if learning rate decay is beneficial for the BaselineNet model on this specific classification task and guide the selection of appropriate decay settings to achieve better convergence and generalization.


# EXPERIMENT 2 DROOUT AND TRASNFER LEARNING
## 2.1 5 DROPOUT RATES

In this experiment, the effect of regularization on the pefromance of the model was test by impoimenting dropout layer and experimenting with the  dropout rate. 

The specification was to impliment it in the fully connected layers of the network. The mothod of implimentation was theough the nn.Dropout module which applies dropout with a probabilty passed as a paramater to the outputs that it is passed (whilst .... the remaining values).

In this arhcitecture of the two fully connected layers one is connected to the output layer so one would not typically apply dropout here (as these connections are outputting the logits which are amount to the classication choice of the network). There is a sense in which, through flattening, the final convolutional layer is fully connected to the first fully connected layer, however it is again not generally recommended to apply dropout to CNN acyivatyions as ... and so it was descided to experiment with a single dropout layer, varying the dropout rate only. This dropout was applued to activations of the 

For this experiment, in accordance with the assignment brief, the original training  data was re-split into 2 halves to create a new dataset for training. 

A set of dropout rates for experimentation were then defined (0, 0.2, 0.4, 0.6, 0.8) and, similar to the previous method for training and evaluating a varying learning rate, 5 identical models were instantiated with different random seeds and trained with all other hyperpapramters being fixed. 

The results were gathers and plotted with averages across the 5 runs for the 5 rates taken in order to compare performance. 

By running this experiment with different dropout rates, the impact of dropout regularization on the model's performance can be observed. The averaged results and smoothed plots provide insights into how the model's training progress, generalization ability, and final performance on the test set are affected by the dropout rate.

The experiment's results can help determine the optimal dropout rate for the DropoutNet model on this specific classification task. It allows for the comparison of different dropout rates and their effect on reducing overfitting and improving generalization. The results can guide the selection of an appropriate dropout rate that achieves the best balance between training performance and generalization.

## 2.2 TRANSFER LEARNING + test data

The second part of this expeiment the experiment aims to investigate the performance of dropout regularization in the context of transfer learning.

It compares the performance of a number of models both in terms of pefrormance during training, but also on the withheld test set. 

The performance of the best performing model from the previous experiments which did not have any opportunity to have any further training on the new data split was compared with 
i) a model trained on the original data without dropout which was been partially retrained on the new data
ii) a model traine don the original data *with** dropout which was partially retrained on the new data. 

In both of the latter cases the partial retaining amounted to transfer learning where some paramtaers learned in the original task were frozen (kept fixed) whereas other were reintialised and made trainable on the new data.

For both of the models with teansfer learning the same process as above was carried out with and average over 5 runs taken for both. 

Performance was compared across training validation and testing. 

The pretrained models are loaded from saved checkpoints using `torch.load()`. The fully connected layers (fc1 and fc2) of both models are modified to match the desired output size of 10 classes.

The experimental settings are defined, including the number of epochs (50), the learning rate (0.1), and a list of random seeds (1 to 5) for reproducibility.

Two models are considered for comparison: model 0 (pretrained_model_non_dropout) and model 1 (pretrained_model_best_dropout). The averaged results for each model will be stored in a dictionary (`averaged_results`).

For each model, the experiment iterates over the random seeds. For each seed, the corresponding pretrained model is loaded and moved to the specified device. The loss function (CrossEntropyLoss) and optimizer (SGD) with the specified learning rate are defined.

The training and validation process is run using the `run_training_and_validation` function, but with a key difference: the training and validation data are swapped. The `swapped_train_dataloader` and `swapped_val_dataloader` are used instead of the original train and validation dataloaders. This is done to evaluate the models' performance on a different data distribution.

The test loss, accuracy, and classification report are obtained by running the `run_testing` function on the trained model using the original test dataloader.

The average train and validation losses and accuracies are calculated across all runs for each model. The average test loss and accuracy are also calculated.

The averaged results for each model are stored in the `averaged_results` dictionary, which includes the random seeds, average train and validation losses and accuracies, individual run results, average test loss and accuracy.

The smoothed train and validation losses and accuracies for each model are plotted using the `plot_single_train_val_smoothed` function, with a smoothing window of 3 and a title indicating the transfer learning model.

If `save_experiment` is set to True, the averaged results are saved to a JSON file specified by `path_to_save`.

By conducting this experiment, the performance of the models with and without dropout regularization can be compared in a transfer learning scenario. The use of swapped train and validation data allows for evaluating the models' ability to generalize to a different data distribution.

The averaged results and smoothed plots provide insights into how the pretrained models with and without dropout perform when fine-tuned on the swapped data. The test results on the original test dataloader assess the models' performance on unseen data.

The experiment's results can help determine the effectiveness of dropout regularization in transfer learning and whether the pretrained model with dropout outperforms the model without dropout in this specific scenario. It provides valuable information on the models' ability to adapt to new data distributions and generalize well.


# EXPERIMENT 3


This experiment investigates the gradient flow in three different neural network models: BaselineNet (without regularization), DropoutNet (with dropout regularization), and BatchNormNet (with batch normalization). The goal was to analyze and compare the mean and standard deviation of the gradients in the first 5 epochs and the last 5 epochs of training for each model.

This was done by extracting the raw gradient values for each layer from the model during training for the first 5 training steps, and the last 5 training steps. 

Conveniently pytorch has this values acessable as aaccessable value in the model. 

# 3.1
this process was done for the baseline model

# 3.2
this process was done for the dropout model

# 3.3
for this experiment it was required to impliment batch normalisation
this process was done for the batch norm model

The experiment follows these steps:

1. Set the number of epochs to 30 and the learning rate to 0.05.

2. For the BaselineNet model:
   - Initialize the model and set the random seed to 1984 for reproducibility.
   - Define the loss function (CrossEntropyLoss) and optimizer (SGD) with the specified learning rate.
   - Collect the absolute gradients for the first 5 epochs and the last 5 epochs using the `collect_gradients_abs` function.
   - Compute the mean and standard deviation of the absolute gradients for the first 5 epochs and the last 5 epochs using the `compute_gradient_statistics_abs` function.
   - Plot the mean and standard deviation of the absolute gradients for the first 5 epochs and the last 5 epochs using the `plot_gradient_statistics_abs` function.

3. Repeat step 2 for the DropoutNet model with a dropout rate of 0.6.

4. Repeat step 2 for the BatchNormNet model.

The `collect_gradients_abs` function collects the absolute gradients for the specified epochs during training. It iterates over the batches in the `train_dataloader` and performs the forward pass, loss computation, and backward pass. The absolute gradients of each layer are collected for the first 5 batches of the first epoch and the last 5 batches of the last epoch.

The `compute_gradient_statistics_abs` function computes the mean and standard deviation of the absolute gradients for each layer based on the collected gradients.

The `plot_gradient_statistics_abs` function visualizes the mean and standard deviation of the absolute gradients for each layer using a bar plot. It creates a figure with two subplots: one for the mean gradients and one for the standard deviations. The x-axis represents the layers, and the y-axis represents the mean or standard deviation values. The bars are grouped by the first 5 epochs and the last 5 epochs for comparison.

By running this experiment, you can observe and compare the gradient flow in the BaselineNet, DropoutNet, and BatchNormNet models. The plots will show the mean and standard deviation of the absolute gradients for each layer in the first 5 epochs and the last 5 epochs. This analysis can provide insights into how the gradients evolve during training and how different regularization techniques (dropout and batch normalization) affect the gradient flow compared to the baseline model.

The results can help understand the impact of regularization on the gradient magnitudes and the stability of the gradient flow throughout the training process.
```

# 3.1 Gradient flow in non-dropout

# 3.2 GRadient flow in dropout

# 3.3 Gradient flow in batch norm no dropout

# 3.4 bacth norm on performance



