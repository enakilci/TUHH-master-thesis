# Emin Çağatay Nakilcioğlu – Master Thesis

## Investigating Artificial Neural Networks (ANN) for Monitoring an Unmanned Air Vehicle


In this thesis, a custom fault injection model for an unmanned air vehicle (UAV) running in
software-in-the-loop (SITL) simulation is proposed together with five distinctive deep learning
based time series prediction model combined with an anomaly detection algorithm to address
anomaly detection problem in a UAV. 

First, SITL simulation of a UAV is performed using a combination of three software tools; an open source control software PX4, an open-source robotics simulator Gazebo and a ground control station software QGroundControl. Via SITL, sensor related data of the simulated UAV is logged and stored. This feature is exploited for applying further analysis on the collected data. Moreover, a custom fault injection (FI) model specifically designed for UAVs running in SITL simulation is developed. Via this FI model, faulty behaviours of a target system are simulated. Combining the SITL mechanism with the FI model provides data source for anomaly detection on a UAV suffering from faults in its system. Data provided by the SITL simulation and FI model consist of 2 different type of datasets;
datasets corresponding to the normal behaviour of the system and datasets corresponding to the behaviour of the system in the presence of any fault in the system. Datasets are further split into two subcategories where for the first category, a UAV follows a circular path in a simulation run whereas second category contains data regarding the UAV following a non-circular path. 

Provided data is utilized as input data for deep learning based models proposed for anomaly detection. Proposed models are mainly categorized in two categories; convolutional neural network (CNN) based and long-short term memory(LSTM) network based models. CNN-based approaches consist of a 1-dimensional CNN model and temporal convolutional network (TCN) whereas LSTM-based approaches are a stacked LSTM model and two LSTM models with two different attention mechanisms. Models are first trained on the dataset with normal behaviour of a UAV with the intention of learning the normal behaviour of a UAV. A secondary dataset containing only normal behaviour of a UAV is used as a validation dataset during training in order to contribute to the generalization ability of the models. During training, models are expected to predict a future data point given the previous related data points. For anomaly detection, a summary prediction model is implemented to the time series predictive models. 

In the anomaly detection algorithm, prediction errors of the models on the training data is computed and later fitted to a multivariate Gaussian distribution. An anomaly score function is derived using the parameters of the Gaussian distribution. Anomaly score is used for classifying whether a given data point is an anomaly point or a normal data point. A fixed threshold of anomaly score for each experiment is individually set. In threshold calculation, anomaly score function is applied to another validation dataset containing both normal and anomalous data points. Anomaly score value yielding the fewer false positives are determined as the threshold value. 

Finally, prediction error of the models on a test dataset containing normal and abnormal points is computed for evaluation of the models’ performance on anomaly detection. Comparison between the performance of the models is made in order to conclude the experiments. The results shows that LSTM-based models yields better performance on flagging detections in a given test dataset as compared with CNN-based models. However CNN-based models provides similar precision values with less computational power.
 
 See masterarbeit_Emin_v3.pdf for the official thesis.

