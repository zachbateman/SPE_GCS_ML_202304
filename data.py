import pandas
from datetime import datetime


def read_input_data(filename: str) -> pandas.DataFrame:
    
    #################################################################################################################
    # READ AND FORMAT DATA ##########################################################################################
    df = pandas.read_csv(filename)
    
    try:  # Remove unallowed input parameter
        df = df.drop(columns=['Avg_PLT_CO2InjRate_TPH'])
    except KeyError:
        pass
    try:  # Remove Training rows without target parameter
        df = df.dropna(axis=0, subset='inj_diff')
    except KeyError:
        pass


    # Turn text into dates for date column. Day/Month in Training data but Month/Day in Test data...
    try:  # D/M/YYYY
        df['SampleTimeUTC'] = df['SampleTimeUTC'].map(lambda text: datetime.strptime(text, '%d/%m/%Y %H:%M'))
    except ValueError: # M/D/YYYY
        df['SampleTimeUTC'] = df['SampleTimeUTC'].map(lambda text: datetime.strptime(text, '%m/%d/%Y %H:%M'))
    
    # Sort all rows from earliest value to latest.
    # This is critical for calulating "DELTA" features which
    # assume each row is right after the previous time.
    df = df.sort_values(by='SampleTimeUTC')




    ####################################################################################################################
    # ADD ADDITIONAL FEATURES BASED ON CHANGES BETWEEN TIME STEPS ######################################################

    # Now add in "DELTA_..." features for the change in values from the previous hour/reading.
    # These may be important as the CHANGE in value from one reading to the next
    # may be more useful than any particular value at one point in time.
    for column in [col for col in df.columns if col not in ['SampleTimeUTC', 'inj_diff']]:
        values = [None]  # can't calculate for first row (no previous value) so assign None
        column_values = df[column].tolist()
        for i in range(1, len(column_values)):
            values.append(column_values[i] - column_values[i-1])
        df[f'DELTA_{column}'] = values

    # Now convert date to a number so it could be used as a regression input
    df['SampleTimeUTC'] = df['SampleTimeUTC'].map(lambda date: (date - datetime(2009, 1, 1)).days)




    #####################################################################################################################
    # CLEAN DATA AS NEEDED ##############################################################################################

    # Annulus pressure has outliers... replace ridiculous values with None (will get replaced by median)
    df['Avg_VW1_ANPs_psi'].values[df['Avg_VW1_ANPs_psi'] > 10_000] = None

    # Lots of zeros in training data target... not sure they add value, so try removing
    try:
        df = df[df['inj_diff'] != 0]
    except KeyError:  # column isn't in testing data
        pass
    
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

    # There seem to be a few huge outliers for inj_diff in training data.
    # Remove these as do not want regression model to attempt to capture
    # these points and potentially mess up the fit on the vast majority of points.
    try:
        CUTOFF = 3000
        df = df[(df['inj_diff'] < CUTOFF) & (df['inj_diff'] > -CUTOFF)]
    except KeyError:  # column isn't in testing data
        pass


    return df
