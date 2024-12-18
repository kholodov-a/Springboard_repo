import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from dateutil import tz
import scipy.stats as st

# Loading timezones for IATA codes of airports
# The source data is 'https://raw.githubusercontent.com/hroptatyr/dateutils/tzmaps/iata.tzmap'
IATAtz_file_path = Path.cwd().parent / 'data' / 'external' / 'iata.tzmap.txt'

IATAtz_df = pd.read_csv(IATAtz_file_path, 
                        sep = '\t', 
                        index_col=0, 
                        header=None)

# Dictionary with IATA codes as keys and timezones as values
IATAtz = IATAtz_df.to_dict('dict')[1]
del(IATAtz_df)

# List of dates of start/end of DST
DST = pd.to_datetime(['2014-03-09', '2014-11-02', '2015-03-08', '2015-11-01', 
       '2016-03-13', '2016-11-06', '2017-03-12', '2017-11-05', 
       '2018-03-11', '2018-11-04'])


def load_data_from(zip_file, data_file, field_type=None):
    '''
    Description
    -----------
    Load data specified as field_type from one data file from zip-archive 

    Parameters
    -----------
    zip_file - path and name of source zip-file contaning 60 csv files
    dat_faile - path and name of csv-file with data
    field_type - dictinary with fields to load and thiers relative data types

    Returns
    -----------
    DataFrame with data loaded
    '''

    # reading the file
    with zipfile.ZipFile(zip_file) as zip_source:
        with zip_source.open(data_file) as file:
            if field_type != None:
                df = pd.read_csv(file, header = 0, 
                                usecols = field_type.keys(),
                                dtype = field_type)
            else:
                df = pd.read_csv(file, header = 0, low_memory=False)

    # Converting dates and boolean        
    if 'FlightDate' in df.columns:
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    if 'DivReachedDest' in df.columns:
        df['DivReachedDest'] = df['DivReachedDest'].fillna(0)
    if 'Cancelled' in df.columns:
        df['Cancelled'] = df['Cancelled'].astype('bool')
    if 'Diverted' in df.columns:
        df['Diverted'] = df['Diverted'].astype('bool')
    if 'DivReachedDest' in df.columns:
        df['DivReachedDest'] = df['DivReachedDest'].astype('bool')
    return df


def time_to_minutes(time_series):
    '''Convert times in HHMM format to minutes
    
    Parameters
    -----------
    time_series     pd.Series   times in HHMM format
    
    Returns
    -----------
    time in menutes from the start of day'''
    return time_series // 100 * 60 + time_series % 100


def get_CRS_datetime(dates, dep_m, arr_m):
    """
    Description
    ------------
    Calculate scheduled(CRS) Departure datetime (timezone naive) based on Flight Date and Departure time (minutes)
    Calculate scheduled(CRS) Arrival datetiem (timezone naive) based on Flight Date and Arrival time (minutes). 
        Arrival time is increased by one day from Fllght Date if the difference between CRS Arrival and Departure time 
        is equal or less than -60 minutes, and decreased by one day if the difference is at least +1380 minutes.

    Parameters:
    ------------
    dates - pandas Series with FlightDate
    dep_m - pandas Seiries with CRS Departure time in minutes 
    arr_m - pandas Seiries with CRS Arrival time in minutes

    Returns:
    ------------
    departure_datetime - CRS Departure datetime (timezone naive) (pd.Series)
    arrival_datetime - CRS Arrival datetime (timezone naive) (pd.Series)
    """

    df = pd.DataFrame() 
    df['departure_datetime'] = pd.to_datetime(dates) + pd.to_timedelta(dep_m, unit='min')

    # Filter for scheduled flights arriving next or privios day
    arr_next_day_filter = arr_m - dep_m  <= -60
    arr_previous_day_filter = arr_m - dep_m  >= 1380

    # For all scheduled flights by default arrival date is eaueal to departure date
    df['arrival_datetime'] = dates

    # Adding one day to the date if flight arrived next day
    df.loc[arr_next_day_filter, 'arrival_datetime'] = df.loc[arr_next_day_filter, 'arrival_datetime'] \
                                                        + datetime.timedelta(1)
    # Adding one day to the date if flight arrived day befor departure day
    df.loc[arr_previous_day_filter, 'arrival_datetime'] = df.loc[arr_previous_day_filter, 'arrival_datetime'] \
                                                        + datetime.timedelta(-1)
    # Finally adding actual arrival time to get actual arrival datetime
    df['arrival_datetime'] = df['arrival_datetime'] + pd.to_timedelta(arr_m, 'm')

    return df['departure_datetime'], df['arrival_datetime']


def get_Actual_datetime(dates, CRSdep_minutes, dep_minutes, arr_minutes, dep_delay):
    """
    Description
    ------------
    Calculate Actual Departure datetime (timezone naive) based on Flight Date, Departure time (minutes) and Departure 
        delay (minutes)
    Calculate Actual Arrival datetiem (timezone naive) based on Flight Date, Actual Departure date in accordance with 
        Departure delay and Arrival time (minutes). Arrival time is increased by one day from Departure Date if 
        the difference between CRS Arrival and Departure time is equal or less than -60 minutes, and decreased by one day 
        if the difference is at least +1380 minutes.

    Parameters:
    ------------
    dates - FlightDate (pd.Series)
    CRSdep_minutes - CRS Departure time in minutes (pd.Series)
    dep_minutes - Actual Departure time in minutes (pd.Series)
    arr_minutes - Actual Arrival time in minutes (pd.Series)
    dep_delay - departure delay in minutes (pd.Series)

    Returns:
    ------------
    departure_datetime - Actual Departure datetime (timezone naive) (pd.Series)
    arrival_datetime - Actual Arrival datetime (timezone naive) (pd.Series)
    """

    # Filters for flights having DepTime_min and ArrTime_min 
    # These filters actually for not cancelled flights, but they differ each other because some flights cancelles after departure
    # So they have departure time but didn't fly 
    ActDep_exists_filter = ~dep_minutes.isna()
    ActArr_exists_filter = ~arr_minutes.isna()

    # Calculating the array with 'day shift' due to flight delay. NOTICE: some flights have -1 day shift because day had a 
    # small negative delay having a scheduled departure time several minutes after midnight
    d = np.zeros([len(dates)], dtype='int')
    d = ((CRSdep_minutes.fillna(0) + dep_delay.fillna(0)) // 1440).astype(int)
    day_deltas_due_to_delay = pd.to_timedelta(d, unit='days')

    # Calculating actual departure datetime
    df = pd.DataFrame() #pd.NaT, index = [*range(0, len(dates))], columns= ['departure_datetime', 'arrival_datetime'])
    df.loc[ActDep_exists_filter, 'departure_datetime'] = dates[ActDep_exists_filter] \
                                                + day_deltas_due_to_delay[ActDep_exists_filter] \
                                                + pd.to_timedelta(dep_minutes[ActDep_exists_filter], 'm')

    # Calculating actual arrival datetime

    # Filter for flights arrived next or previous day
    Arrived_next_day = arr_minutes - dep_minutes  <= -60
    Arrived_previous_day = arr_minutes - dep_minutes  >= 1380

    # For all arrived flights at first arrival date is eaueal to departure date
    df.loc[ActArr_exists_filter, 'arrival_datetime'] = dates[ActArr_exists_filter] \
                                                + day_deltas_due_to_delay[ActArr_exists_filter] 
    # Adding one day to the date if flight arrived next day
    df.loc[ActArr_exists_filter & Arrived_next_day, 'arrival_datetime'] = \
            df.loc[ActArr_exists_filter & Arrived_next_day, 'arrival_datetime'] + datetime.timedelta(1)
    
    # Adding one day to the date if flight arrived previous day
    df.loc[ActArr_exists_filter & Arrived_previous_day, 'arrival_datetime'] = \
            df.loc[ActArr_exists_filter & Arrived_previous_day, 'arrival_datetime'] + datetime.timedelta(-1)
    
    # Finally adding actual arrival time to get actual arrival datetime
    df.loc[ActArr_exists_filter, 'arrival_datetime'] = \
            df.loc[ActArr_exists_filter, 'arrival_datetime'] + pd.to_timedelta(arr_minutes.loc[ActArr_exists_filter], 'm')
    
    return df['departure_datetime'], df['arrival_datetime']


def convert_column_to_UTC(df, dt_field, IATA_code_field):
    '''Convert a column with datetimes timezone naive to the column with UTC timezone

    Parameters:
    ------------
    df              DataFrame   The dataframe containing the field (column) to be converted
    df_field        string      The name of the column in the dataset to convert from
    IATA_code_field string      The name of the column in the dataset with IATA codes of airports to get the time zone of the airport

    Returns:
    ------------
    A list with datetimes in UTC timezone
    '''
    return [row[dt_field].tz_localize(tz=tz.gettz(IATAtz[row[IATA_code_field]]), ambiguous=True, nonexistent='shift_forward')
             .astimezone(tz.UTC) for _, row in df.iterrows()]

def local_to_UTC(dt_naive, IATA_code):
    '''Converts datetime timezone naive value to UTC timezone datetime
    '''
    return dt_naive.tz_localize(tz=tz.gettz(IATAtz[IATA_code]), ambiguous=True, nonexistent='shift_forward').astimezone(tz.UTC)


def UTC_to_local(dt_UTC, IATA_code):
    '''Converts UTC datetime to timezone naive datetime
    '''
    return dt_UTC.astimezone(tz.gettz(IATAtz[IATA_code])).replace(tzinfo=None)

def naive_to_tz_aware(dt_naive, IATA_code):
    '''Converting datetime tz-naive value to tz-aware local time
    '''
    return dt_naive.tz_localize(tz=tz.gettz(IATAtz[IATA_code]), ambiguous=True, nonexistent='shift_forward')


def add_local_tz(df, dt_field, IATA_code_field):
    '''Function to add to the series of datetime a timezone specified by airports IATA codes
    
    Papameters
    ------------
    df                  DataFrame   - dataset with flights records
    dt_field            str         - name of the field with tz-naive datetime
    IATA_code_filed     str         - name of the field with IATA airport code
    '''
    return [row[dt_field].tz_localize(tz=tz.gettz(IATAtz[row[IATA_code_field]]), ambiguous=True ,nonexistent='shift_forward')
        for _, row in df.iterrows()]


def memory_usage_per_type(data_frame):
    '''Prints the usage of memory by the DataFrame per each type of data

    Papameters
    ------------
    data_frame  pd.DataFrame

    Returns
    ------------    
    Prints the memory size allocated to each data type in the DataFrame

    Source: https://medium.com/@alielagrebi/optimize-the-pandas-dataframe-memory-consuming-for-low-environment-24aa74cf9413
    '''
    types = ['number', 'object', 'datetimetz', 'category', 'bool']
    for tp in types:
        selected_col = data_frame.select_dtypes(include=[tp])
        memory_usage_per_type_b = selected_col.memory_usage(deep=True).sum()
        memory_usage_per_type_mb = memory_usage_per_type_b / 1024**2
        print('memory usage for {} columns: {:03.3f} MB'.format(tp, memory_usage_per_type_mb))
    

def memory_usage(data_frame):
    '''Returns the memory size allocated to the entire DataFrame

    Papameters
    ------------
    data_frame  pd.DataFrame

    Returns
    ------------    
    The memory size allocated to the entire DataFrame
    
    Source: https://medium.com/@alielagrebi/optimize-the-pandas-dataframe-memory-consuming-for-low-environment-24aa74cf9413
    '''
    
    return data_frame.memory_usage(deep=True).sum() / 1024**2


def isDST(ds, field_name_1, field_name_2):
    '''Returns the sliice of dataset where at least one of two fields of a row is DST start or finish date
    
    Papameters
    ------------
    ds              pd.DateFrame    a dataframe with flights data
    field_name_1    str             the name of the column with datetime (the departure date)
    field_name_2    str             the name of the column with datetime (the arrival date)

    Returns
    ------------
    Returns the dataframe where at least one of the dates (departure or/and errival) is the date of DST start of finish
    '''
    return (pd.to_datetime(ds.loc[:, field_name_1].dt.date).isin(DST) 
            | pd.to_datetime(ds.loc[:,field_name_2].dt.date).isin(DST))


def correct_IDL(df, time_type):
    '''
    Description
    ------------
    Correcting time data for HML and GUM airports which are across IDL 

    Parameters:
    ------------
    df - flights dataset (pd.DateFrame)
    time_type - 'CRS' for scheduled times and 'Act' for actual
    
    Returns:
    ------------
    -1      if the type of the datetime column for correction is wrong
    '''
    if time_type not in ['CRS', 'Act']:
        print('Wrong parameter:', time_type)
        return -1
    
    diff = 'diff_' + time_type
    arrUTC = time_type + 'Arr_UTC'
    UTCElapsed = 'UTCElapsedTime_' + time_type

    error_1440_filter = df[diff].isin([1440, -1440])
    error_1440 = df[error_1440_filter]
    HNL_GUM_filter = (error_1440['Origin'].isin(['HNL', 'GUM'])) & (error_1440['Dest'].isin(['HNL', 'GUM']))
    error_1440 = error_1440[HNL_GUM_filter]
    print(error_1440[['Origin', 'Dest', 
                'CRSDepDT', 'CRSArrDT',
                'CRSDep_UTC', 'CRSArr_UTC', 
                'CRSElapsedTime', 'UTCElapsedTime_CRS', 'diff_CRS']])
    error_1440[diff].astype(float)
    error_1440[arrUTC] -= pd.to_timedelta(error_1440[diff], 'min')
    error_1440[UTCElapsed] -= error_1440[diff]
    error_1440[diff] = 0
    print(error_1440[['Origin', 'Dest', 
                'CRSDepDT', 'CRSArrDT',
                'CRSDep_UTC', 'CRSArr_UTC', 
                'CRSElapsedTime', 'UTCElapsedTime_CRS', 'diff_CRS']])

    df.loc[error_1440_filter, diff] = error_1440[diff]
    df.loc[error_1440_filter, arrUTC] = error_1440[arrUTC]
    df.loc[error_1440_filter, UTCElapsed] = error_1440[UTCElapsed] 
    # df.loc[error_1440_filter, [diff + arrUTC + UTCElapsed]] = error_1440[diff + arrUTC + UTCElapsed]


def correct_error_of_time(df, time_type, threshold):
    '''
    Description
    ------------
    Correcting the Elapsed Times errors in the dataset by subsetting the simillar flights (the same Origin and Destination) 
    and comparing the flight's Elapsed times (Actual and CRS) with the actual Elapsed time of similar flights 

    Parameters:
    ------------
    df          pd.DataFrame    flights dataset (pd.DateFrame)
    time_type   str             'CRS' for scheduled times and 'Act' for actual
    threshold   float           the theashold for ratio of error to median value for similar flights 
    
    Returns:
    ------------
    -1      if the type of the datetime column for correction is wrong
    '''

    if time_type not in ['CRS', 'Act']:
        print('Wrong parameter:', time_type)
        return -1

    diff = 'diff_' + time_type
    arrUTC = time_type + 'Arr_UTC'
    depUTC = time_type + 'Dep_UTC'
    UTCElapsed = 'UTCElapsedTime_' + time_type
    if time_type == 'Act':
        ElapsedTime = 'ActualElapsedTime'
        arrDT = 'ArrDT'
        depDT = 'DepDT'
    else:
        ElapsedTime = 'CRSElapsedTime'
        arrDT = time_type + 'ArrDT'
        depDT = time_type + 'DepDT'


    # Process only records that have non-Zero and non-NaN difference
    differencies = df[(df[diff] != 0) & (~df[diff].isna())]
    
    total_number = len(differencies)
    counter_corrected = 0
    counter_too_far = 0
    counter_unique = 0

    for ind, row in differencies.iterrows():
        # Finding flights for the same Origin and Destination to find an estimate of flight durations (ET)
        similar_df_filter = (df['Origin'] == row['Origin']) \
                                & (df['Dest'] == row['Dest'])
        similar_df = df[similar_df_filter][ElapsedTime]
        similar_df = similar_df[similar_df.notna()]
        similar_df = similar_df.astype(int)

        # Only if similar fligths were found
        if (len(similar_df) - 1) > 0:
            # Confidence interval 
            min_ci_time, max_ci_time = st.t.interval(0.95, len(similar_df)-1, 
                                                        loc = np.mean(similar_df), 
                                                        scale = st.sem(similar_df))
            
            mid_time = (min_ci_time + max_ci_time) / 2

            ET_from_mid_time = abs(row[ElapsedTime] - mid_time)
            UTC_from_mid_time = abs(row[UTCElapsed] - mid_time)

            # if (ET_from_mid_time < UTC_from_mid_time) & ((ET_from_mid_time / mid_time) < threshold):   # Sourced ET is closer to mediad time
            #     # df.loc[ind, arrUTC] = local_to_UTC(row[arrDT], row['Dest'])
            #     # df.loc[ind, UTCElapsed] = (df.loc[ind, arrUTC] - df.loc[ind, depUTC]).total_seconds()/60
            #     # df.loc[ind, diff] = (df.loc[ind, UTCElapsed] - df.loc[ind, ElapsedTime])
            #     counter_utc_based += 1
            if ((UTC_from_mid_time / mid_time)  < threshold): # UTC-based ET is close to median timm
                df.loc[ind, arrDT] = UTC_to_local(row[arrUTC], row['Dest'])
                df.loc[ind, ElapsedTime] = (df.loc[ind, arrUTC] - df.loc[ind, depUTC]).total_seconds()/60
                df.loc[ind, diff] = (df.loc[ind, UTCElapsed] - df.loc[ind, ElapsedTime])
                counter_corrected += 1
            else:
                print('The flight\'s (index {:d}) UTC elapsed times is too far from median time.'.format(ind))
                print('Confidence interval ({:.1f}, {:.1f}) with {:s} ET =  {:.1f} min and UTC ET = {:.1f} min'.format(min_ci_time, 
                                                                                                                       max_ci_time, 
                                                                                                                       time_type,
                                                                                                                       row[ElapsedTime], 
                                                                                                                       row[UTCElapsed]))
                counter_too_far += 1
        else:
            print('The flight number {:d} from {:s} to {:s} on {:%Y-%m-%d %H:%M} is unique (ind: {:d})'
                  .format(row['Flight_Number_Reporting_Airline'], row['Origin'], row['Dest'], row['CRSDepDT'], ind))
            counter_unique += 1

    print('__________'*8)
    print('Total number of cases:', total_number )
    print('      {:d} of them were corrected'.format(counter_corrected))
    print('      {:d} flights have incorrect departure or arrival time (ambiguous to fix the error)'.format(counter_too_far))
    print('      {:d} flights don\'t have enought simillar flights to compare with'.format(counter_unique))
    print('\n\n\n')


def get_outliers_range(values):
    ''' Returns the range for ouliers as 25th quantile - 1.5 * IQR adn 75th quantile + 1.5 * IQR
    '''
    q25, q75 = values.quantile([0.25, 0.75])
    return q25 - (q75 - q25) * 1.5, q75 + (q75 - q25) * 1.5