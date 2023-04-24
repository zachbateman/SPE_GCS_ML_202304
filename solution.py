import evogression
import data


df = data.read_input_data('illinois_basing_train.csv')


# sample = df.sample(5000)
# evogression.generate_robust_param_usage_file('inj_diff', sample, num_models=30, creatures=15_000, cycles=12)
# breakpoint()


high_graded_features = [
    'DELTA_Avg_CCS1_DH6325Ps_psi',
    'DELTA_Avg_PLT_CO2VentRate_TPH',
    'Avg_VW1_Z08D5840Tp_F',
    'DELTA_Avg_CCS1_DH6325Tp_F',
    'Avg_VW1_Z02D6982Ps_psi',
    'Avg_CCS1_DH6325Ps_psi',
]
df = df[['inj_diff'] + high_graded_features]


model = evogression.Evolution('inj_diff', df, creatures=100_000, cycles=15)
model.output_regression()
df = model.predict(df)
df.to_excel('Predicted_new.xlsx')


test_df = data.read_input_data('illinois_basing_test_04112023.csv', test_data=True)
test_df = model.predict(test_df, prediction_key='inj_diff')
test_df = test_df[['inj_diff']]
test_df.to_csv('SUBMISSION.csv', index=False)
