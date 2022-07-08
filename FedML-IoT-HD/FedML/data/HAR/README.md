# Human Activity Recognition Using Smartphones Data Set

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

**Dataset Characteristics:** Multivariate, Time-series

**Tasks:** Classification, clustering, etc.

**Number of instances:** 10299

**Number of Attributes:** 561

**Number of subjects:** 30

**Cite:** 

```
@inproceedings{anguita2013public,
  title={A public domain dataset for human activity recognition using smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra Perez, Xavier and Reyes Ortiz, Jorge Luis},
  booktitle={Proceedings of the 21th international European symposium on artificial neural networks, computational intelligence and machine learning},
  pages={437--442},
  year={2013}
}
```

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

## Getting Started

* Download the dataset from UCI's repository and unzip `UCI HAR Dataset.zip` in this directory
* Run `python3 read_data.py --tf 0.9` to import data and split data into training and testing, and store the data into `train.json` and `test.json` format. The stored json format is the same as the [LEAF](https://leaf.cmu.edu/) dataset in preparation for Federated Learning applications.
  * `--tf` := fraction of data in training set, written as a decimal; default is 0.9


