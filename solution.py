import evogression
import data


df = data.read_input_data('illinois_basin_train.csv')

# Filter down dataset to only the features that were determined to be
# most predictive by the output file from feature_high_grading.py.
# It should not be strictly necessary to cull down the features, but
# doing so makes it easier to find a good regression model.
high_graded_features = [
    'DELTA_Avg_CCS1_DH6325Ps_psi',
    'DELTA_Avg_PLT_CO2VentRate_TPH',
    'DELTA_Avg_CCS1_WHCO2InjTp_F',
    'DELTA_Avg_CCS1_DH6325Tp_F',
]

df = df[['inj_diff'] + high_graded_features]

# Train a regression model on input data with high-graded features.
# "creatures" and "cycles" kwargs indicate how hard the ML algorithm searches for a fitting equation.
# Originally used creatures=300_000 and cycles=25, but these smaller values run faster with similar accuracy.
# "creatures" is the number of randomly-generated equations attempting to predict 'inj_diff'.
# "cycles" is the number of cycles run for mutating those equations and trying new ones.
model = evogression.Evolution('inj_diff', df, creatures=25_000, cycles=15)

# Output the regression function to a Python module for reference.
model.output_regression(add_error_value=True)

# Output the Training data with predictions added for reference.
df = model.predict(df, prediction_key='inj_diff_PREDICTED')
df.to_excel('Training_With_Predictions.xlsx')


# Using trained model, generate submission file from Test data
test_df = data.read_input_data('illinois_basin_test_04112023.csv')
test_df = model.predict(test_df, prediction_key='inj_diff')  # Add predictions
test_df = test_df[['inj_diff']]  # Filter to only have prediction column
test_df.to_csv('SUBMISSION.csv', index=False)  # Output Submission CSV file
