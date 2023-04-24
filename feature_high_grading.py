import evogression
import data


df = data.read_input_data('illinois_basing_train.csv')

# Sample the data to cut down on time needed to run analysis.
# This should still be a sufficiently large sample to get representative feature usefulness.
sample = df.sample(7500)

# This function runs through many iterations of attempting to predict "inj_diff"
# using various combinations of all the input parameters.
# It then outputs an Excel file ("RobustParameterUsage.xlsx") which
# ranks the parameters from most to least useful at helping make predictions.
evogression.generate_robust_param_usage_file('inj_diff', sample, num_models=30, creatures=15_000, cycles=10)
