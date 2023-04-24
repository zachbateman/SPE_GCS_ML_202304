import pandas
from datetime import datetime



def read_input_data(filename, test_data: bool=False):
    df = pandas.read_csv(filename)
    try:
        df = df.drop(columns=['Avg_PLT_CO2InjRate_TPH'])
    except KeyError:
        pass
    try:
        df = df.dropna(axis=0, subset='inj_diff')
    except KeyError:
        pass

    date_format = '%d/%m/%Y %H:%M' if not test_data else '%m/%d/%Y %H:%M'

    # Turn text into dates for date column
    df['SampleTimeUTC'] = df['SampleTimeUTC'].map(lambda text: datetime.strptime(text, date_format))
    df = df.sort_values(by='SampleTimeUTC')

    # # Add in rough atmospheric temperature feature per provided paper
    month_temp = {1: -2, 2: 0, 3: 4, 4: 10, 5: 16, 6: 23, 7: 24, 8: 21, 9: 15, 10: 7, 11: 3, 12: -1}
    df['MonthlyTemp'] = df['SampleTimeUTC'].map(lambda date: month_temp[date.month])

    # # Add in average monthly precipitation feature per provided paper
    month_precipation = {1: 44, 2: 50, 3: 67, 4: 97, 5: 109, 6: 110, 7: 98, 8: 93, 9: 73, 10: 83, 11: 87, 12: 65}
    df['Precipitation'] = df['SampleTimeUTC'].map(lambda date: month_precipation[date.month])

    # Now convert date to a number so can be used in regression
    df['SampleTimeUTC'] = df['SampleTimeUTC'].map(lambda date: (date - datetime(2009, 1, 1)).days)


    df['WellborePressureDiff'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']
    df['WellboreTempDiff'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']
        
    # Annulus pressure has outliers... replace rediculous values with None
    df['Avg_VW1_ANPs_psi'].values[df['Avg_VW1_ANPs_psi'] > 10_000] = None

    # Lots of zeros in training data... not sure they add value, so try removing
    try:
        df = df[df['inj_diff'] != 0]
    except KeyError:
        pass

    # Now add in features for the change in values from the previous hour
    for col in [col for col in df.columns if col not in ['SampleTimeUTC', 'MonthlyTemp', 'Precipitation', 'inj_diff']]:
        values = [None]  # can't calculate for first row so assign None
        base_values = df[col].tolist()
        for i in range(1, len(base_values)):
            values.append(base_values[i] - base_values[i-1])
        df[f'DELTA_{col}'] = values

    # Have many zero readings for temp and pressure, but these readings should never be zero...
    # Replace with None which then gets assigned median value in regression
    data = df.to_dict('records')
    cleaned_data = []
    for row in data:
        new_row = {}
        for column, value in row.items():
            if value == 0 and ('PSI' in column.upper() or 'TP_F' in column.upper()):
                new_val = None
            else:
                new_val = value
            new_row[column] = new_val
        cleaned_data.append(new_row)
    df = pandas.DataFrame(cleaned_data)

    try:
        CUTOFF = 3000
        # outliers = df[(df['inj_diff'] < -CUTOFF) | (df['inj_diff'] > CUTOFF)]
        df = df[(df['inj_diff'] < CUTOFF) & (df['inj_diff'] > -CUTOFF)]
    except KeyError:
        pass
    
    return df