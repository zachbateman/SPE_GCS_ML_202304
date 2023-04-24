# SPE-GCS April 2023 Machine Learning Challenge

This repository is a submission for the SPE Gulf Coast Section April/May 2023 Machine Learning Challenge - Using AI to Validate Carbon Containment in the Illinois Basin.  

# Project Overview

This Machine Learning Challenge focuses on the Illinois Basin – Decatur Project which is a CO2 storage project using the Mount Simon Sandstone.

The objective of the challenge is to predict the “inj_diff” value given an assortment of possible input parameters mostly consisting of fiber-optic wellbore pressure and temperature readings.

The data is provided as measurement readings with specific timestamps with one hour in between readings.  Assuming the target “inj_diff” column can be estimated from the provided input data, a Machine Learning algorithm should be able to find which inputs are useful and how to use them.


# Data Exploration

Possible input paramters/features are mostly pressure and temperature readings from two wellbores with many readings from different depths.  Attempts to predict the target ("inj_diff") from the raw data were initially unpromising.

Given that the project and the target variable seem to be interested in how injection is changing, additional "DELTA_..." features were added specifying the change in each value between timestamps.

`data.py` contains the logic for reading, cleaning, and adding new features to the input data files (both Train and Test).


# High-Graded Input Parameters

After adding “DELTA_...” features based on the change between timesteps for each column, each of the possible inputs were evaluated for their ability to predict “inj_diff” using `feature_high_grading.py`.

The DELTA values turned out to be far more useful than the raw values of pressure, temperature, etc.

Four input features were selected as the most predictive attributes to use for the final Machine Learning model:
 - DELTA_Avg_CCS1_DH6325Ps_psi
 - DELTA_Avg_PLT_CO2VentRate_TPH
 - DELTA_Avg_CCS1_WHCO2InjTp_F
 - DELTA_Avg_CCS1_DH6325Tp_F


# Final Solution

The four high-graded input parameters above were combined with the target "inj_diff" column in the training data.  This smaller version of training data was then used to train the ML model and add predictions to the test data using `solution.py`.  The model used for the initial test submission was saved as `regression_function.py`, but `solution.py` will build a new (but similar) model when run again for final testing.


License
----
Apache 2.0