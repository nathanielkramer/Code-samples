##########################################################################################################################################

#	This script was created as a step in a program to forecast workflow volumes for call center employees.
#	The particular forecast models used were chosen in a separate analysis performed in a jupyter notebook file.
#	(An example of this work can be seen in this repository here: https://github.com/nathanielkramer/Code-samples/blob/main/Python-samples/NAT-850%20Erlang%20Forecast-Implementation%20notebook.ipynb)

#	This script connects to a SQL database, executes a query to pull the most recent weekly volume data into a pandas dataframe,
#	converts this data to a darts timeseries object, and then forecasts volume for the next two weeks. Then, these weekly forecasts
#	are converted to half hourly-level forecasts by calculating the average distribution of weekly volume across each half-hour period
#	for the previous 8 weeks, using Z-score to exclude outliers and avoid skewing the distribution. Finally, this distribution is applied
# 	to the weekly-level forecast volume to convert it to a half hourly level-forecast. The final step in the script exports the forecast
#	to SQL server and executes a stored procedure in the SQL database to perform further processing on the output data.

##########################################################################################################################################



# ### Selected Forecast Models
# Model selection should be verified periodically to ensure that we are still 
# using the best models for capturing forecast volume in calls and chats. 
# Current models are:

# Chats: ARIMA(1,1,1)x(1,0,0)[52]   
# Calls: ARIMA(0,1,3)x(1,0,0)[52]




# Import necessary libraries
import pandas as pd
import numpy as np

# forecasting library and sub-libraries
import darts
import darts.models as mod

# for filtering data points in bucketing step
from scipy import stats

# to create SQL connection
from sqlalchemy import create_engine


############
# create SQL connection to the network services db here
server = 'CMMSQLRPT2'
db = 'network_services'
driver = 'SQL+Server'
conn_str = 'mssql+pyodbc://{}/{}?driver={}'.format(server,db,driver)
engine = create_engine(conn_str)


############
# declare important program variables

# instantiate our models (these should be checked and revised periodically for optimal performance)
chat_model = mod.ARIMA(p=1,d=1,q=1, seasonal_order=(1,0,0,52))
call_model = mod.ARIMA(p=0,d=1,q=3, seasonal_order=(1,0,0,52))

# name of output table
SQL_output_table = 'tmp_wfm_erlang_inputs_v2_NAK' # 't_wfm_erlang_inputs'

# name of proc
SQL_erlang_proc = 'TMP_NAK_p_wfm_erlang_input_v2' # 'p_wfm_erlang_input'


############
# function definitions

# function to forecast two steps ahead for Erlang model
def erlang_forecast(model, input_data, forecast_horizon=2, return_model = False):
    '''
    A more targeted version of the fit_model function. This simply takes a time series (presumably call or chat data)
    and returns an n-steps-ahead forecast. No plotting, no MAPE, nada.
    
    There is a toggle to allow you to return both the forecast and the fitted model, or just the forecast. If return_model
    is selected, it will return a dictionary indexed with 'model' and 'forecast'
    '''
    # fit the model and run the forecast for the selected time horizon
    model.fit(input_data)
    forecast = model.predict(forecast_horizon)
    
    # return the selected results
    if return_model:
        return {'model': model, 'forecast':forecast}
    else:
        return forecast


# function to combine timeseries objects into a dataframe
def combine_TS(ts_list):
    '''
    Pass an iterable of time series with the same axes, this will combine them
    '''
    # initiate a dataframe from the first timeseries
    df = ts_list[0].pd_dataframe()
    
    # append each remaining timeseries to the dataframe
    for ts in ts_list[1:]:
        df = df.merge(ts.pd_dataframe(), how='outer', left_index=True, right_index=True)
        
    # drop the name field of the df (default behavior of TS.pd_dataframe() names the columns "component")
    df.columns.name = None
    
    return df


# function to calculate z-scores and mark threshold
def bucket_z_scores(df, z_threshold):
    '''This function applies the method used in NAT-850 to calculate z-scores for each combination of DoW/hour in the data
    , and then detemerine whether each data point is an outlier that should be dropped. 
    '''
    # calculate the z scores
    df['grouped z-score'] = df.groupby(['interaction_type','dayname','hour'])['volume'].transform(lambda x : stats.zscore(x))
    
    # determine if the z-scores exceed our threshold value (2-tailed, so use abs)
    df['exceeds z-score threshold'] = abs(df['grouped z-score']) > z_threshold
    
    return None


# function to execute a SQL stored procedure
def exec_stored_procedure(engine, proc_name):
    '''Pass an active SQLAlchemy Engine object and the name of a proc to this function and it will execute the proc.'''
    
    with engine.begin() as conn:
        conn.execute(f"EXEC {proc_name}")
        
    return None


############
# get the data
# NOTE: this query has been slightly modified (the final WHERE clause) to ensure replicability if this program is run later 

# query params (can use this to filter dates for testing if desired):
param = 'DATEADD(ww,DATEDIFF(ww,0,getdate()),-1)' # drop data from the current (partial) week

query = f'''
--------------------------------------
-- get initial call data
with get_calls as (
	select call_id
		, call_datetime
		,cast(call_datetime as date) as date
		,DATEADD(mi,
			(DATEPART(mi, call_datetime)/30)*30,
			DATEADD(hour,datediff(hour,0,call_datetime),0)) as bucket
	from t_ns_calls
	where call_type = 'Inbound'
		and (
			(datepart(dw, call_datetime) = 7
			and datepart(hh, call_datetime) between 8 and 17)
		or  (datepart(dw, call_datetime) between 2 and 6
			and datepart(hh, call_datetime) between 8 and 22)
			)
	)

-- total call data by bucket
, call_buckets as (
	select
		'Actual' as series
		,DATEADD(ww,DATEDIFF(ww,0,bucket),-1) as weekstart --use -1 as the starting point since date 0 is a Monday (so this will give us weekstart as Sunday)
		,max(date) as date
		,bucket
		,'Call' as interaction_type
		,DATEPART(dw, bucket) as day_of_week
		,COUNT(distinct call_id) as n_calls
	from get_calls
	group by bucket
	)

--------------------------------------
-- get initial chat data
, get_chats as (
	select 
		chat_id
		,conversation_sequence
		,conversation_start_time
		,cast(conversation_start_time as date) as date
		,DATEADD(mi,
			(DATEPART(mi, conversation_start_time)/30)*30,
			DATEADD(hour,datediff(hour,0,conversation_start_time),0)) as bucket
	from t_ns_chats
	where num_coord_messages > 0
		and conversation_start_time > '2020-01-06 00:00:00.000' -- added to filter the initial partial week of data
		and (
			(datepart(dw, conversation_start_time) = 7
			and datepart(hh, conversation_start_time) between 8 and 17)
		or  (datepart(dw, conversation_start_time) between 2 and 6
			and datepart(hh, conversation_start_time) between 8 and 22)
			)
	)

-- total chat data by bucket
, chat_buckets as (
	select
		'Actual' as series
		,DATEADD(ww,DATEDIFF(ww,0,bucket),-1) as weekstart --use -1 as the starting point since date 0 is a Monday (so this will give us weekstart as Sunday)
		,max(date) as date
		,bucket
		,'Chat' as interaction_type
		,DATEPART(dw, bucket) as day_of_week
		,COUNT(chat_id) as n_chats
	from get_chats
	group by bucket
	)

-- get an accurate holiday calendar to add in
, calendar as (
	select 
		date_val
		, weekday_name as dayname
		, CAST(CASE WHEN (isFederalHoliday=1 AND is_holiday=1) 
					THEN 0 
					ELSE is_workday 
				END as BIT) as is_workday
		, CAST(CASE WHEN (isFederalHoliday=1 AND is_holiday=1) 
					  or (calendar_year=2020 AND is_holiday=1 AND is_weekday = 1 
						AND date_val NOT IN ('2020-11-27','2020-12-24')) -- the OR part accounts for missing federal holiday data prior to 2021 (with hard-code for Friday after thanksgiving)
					THEN 1 
					ELSE 0 
				END as BIT) as is_holiday 
	from
		dw_common.dbo.dim_date
	where date_val > '2020-01-01' -- no call/chat data before 2020
		and date_val <= dateadd(yy,1,getdate())
	)


-------------------------------------------------------------------------------------------
---------------------------------- END CTE TABLES -----------------------------------------
-------------------------------------------------------------------------------------------

-- Single query to combine all
select t1.*
	,t2.is_holiday
	,t2.is_workday
	,t2.dayname
from (
	select
		weekstart
		,date
		,bucket
		,day_of_week
		,interaction_type
		,n_calls as volume
		from call_buckets
	UNION ALL
	select
		weekstart
		,date
		,bucket
		,day_of_week
		,interaction_type
		,n_chats as volume
		from chat_buckets
	) t1
left join calendar t2
on t1. date = t2.date_val
where t1.weekstart < {param}
order by t1.bucket
'''

# execute the query above to get our input data
raw_df = pd.read_sql(query,engine)


# total historical volume by week
keepcols = ['weekstart','interaction_type','volume']
weekly_df = raw_df[keepcols].groupby(['weekstart','interaction_type']).sum().reset_index()

# also count workdays and holidays
day_counts = raw_df[['weekstart','date','is_holiday','is_workday']].groupby(['weekstart','date']).max().reset_index().groupby('weekstart').sum(numeric_only=True)

# add the count of workdays and holidays to weekly_df
weekly_df = weekly_df.merge(day_counts, left_on='weekstart', how='left', right_index=True)

# reformat the data
vol_df = weekly_df[['weekstart','interaction_type','volume','is_holiday']].copy()
vol_df = vol_df.pivot(columns='interaction_type', values='volume',index='weekstart')

# reset the name field of the columns because it causes darts ts conversion to fail
vol_df.columns.name=None


# convert chat to 2 column TS for forecasting in darts
TS = darts.TimeSeries.from_dataframe(
        vol_df.merge(day_counts[['is_holiday']], right_index=True, left_index=True, how='left')
        )


##########
# ## Run the forecast models and rearrange the outputs

# Fit the models assigned at the top of this program.
# Then, use our forecast function to execute a forecast.
# Next, use combine function to combine the forecast outputs


# fit the models
# Reminder of function args: erlang_forecast(model, input_data, forecast_horizon=2, return_model = False)
chat_forecast = erlang_forecast(chat_model, TS['Chat'])
call_forecast = erlang_forecast(call_model, TS['Call'])


# output to a dataframe
forecast_df = combine_TS([chat_forecast,call_forecast])

# add week number to match the desired SQL output
forecast_df.insert(0, 'forecast_week', range(1,1+len(forecast_df))) 


###########
# ## Part 2: bucketize the data
# The above portion successfully runs our forecast and spits it out as a pandas dataframe. The next step is to
# get an average distribution of volume by day and time, which is below. Then, we'll put it all together--multiply
# the forecast by the buckets to get our bucketized forecast.  

# get the last 8 weeks for averaging
tgt_date = raw_df['weekstart'].max() - pd.Timedelta(8,'W')

block_df = raw_df[raw_df['weekstart'] > tgt_date].copy().reset_index(drop=True)

# add in an 'hour' column for use when averaging later
block_df['hour'] = block_df['bucket'].dt.strftime('%H:%M')

# ignore holidays as they may skew the percentages
block_df = block_df[~ block_df['is_holiday']]


# ## Part 2A: Filter outliers
# Using the z-score methodology developed in NAT-850, filter outliers from the data before we average

# use the function defined at the top of the program to create the filter column
bucket_z_scores(block_df, 2)

# drop the data points that exceed the z-score threshold from the data
block_df = block_df[~block_df['exceeds z-score threshold']]


# ## Back to the regular bucketizing steps:
    
# Adjust the weekly volume to account for short weeks
# first, let's grab only the period of interest
adj_weekly_df = weekly_df[weekly_df['weekstart'] > tgt_date].copy()

# for weeks with 4 workdays, multiply volumes by a factor of 5/4
adj_weekly_df['adj_weekly_vol'] = adj_weekly_df['volume'] * (5 / (5 - adj_weekly_df['is_holiday']))

# join the weekly volumes to the block df
block_df = block_df.merge(adj_weekly_df[['weekstart','interaction_type','adj_weekly_vol']], on=['weekstart','interaction_type'], how='left')

# calculate the percentage of total weekly volume that occurs in each block
block_df['volume_pct'] = block_df['volume'] / block_df['adj_weekly_vol']

# calculate averages for each day/time period
group_cols = ['dayname','day_of_week','hour','interaction_type']
weekly_avgs = block_df[group_cols + ['volume_pct']].groupby(group_cols, as_index=False).mean()


# adjust the percentages so that the averages all total to 100
# first, calculate current totals
interaction_totals = weekly_avgs.groupby('interaction_type', as_index=False).sum(numeric_only=True)
weekly_avgs = weekly_avgs.merge(interaction_totals[['interaction_type','volume_pct']].rename(columns={'volume_pct':'adj factor'}), on ='interaction_type')

# then, divide each observation by the total to get them to add to 100
weekly_avgs['adj_vol_pct'] = weekly_avgs['volume_pct'] / weekly_avgs['adj factor']

weekly_avgs.drop(columns='adj factor', inplace=True)


# rearrange the data and add an interaction_type column (in one line using list comprehension)
forecast_rearrange = pd.concat(
    [forecast_df[['forecast_week',item]].rename(columns={item:'Volume'}).assign(interaction_type=item) 
     for item in ['Chat','Call']
     ]
    ).reset_index()


# merge the forecast output with the average volume percentages
output_df = forecast_rearrange.merge(weekly_avgs, on='interaction_type', how='left')

# calculate the volume by block 
output_df['average'] = output_df['Volume'] * output_df['adj_vol_pct']

# drop unnecessary columns from the output
drop_cols=['Volume', 'volume_pct', 'adj_vol_pct']
output_df.drop(columns=drop_cols, inplace=True)

# add a 'category' column to match t_wfm_erlang_inputs
output_df['category'] = 'Volume'


#################
# # Final step: outputting to SQL
# Reformat the output data, add an empty column for rundate, output, and then 
# execute the p_wfm_erlang_input stored procedure.

# reorganize the output columns and add rundate
col_order = ['run_date', 'category', 'interaction_type', 'forecast_week', 'day_of_week', 'hour', 'average']

# add a NULL placeholder value for run_date, which will be updated in the SQL proc
output_df['run_date'] = np.NaN

# output the volume forecast data to SQL
output_df[col_order].to_sql(SQL_output_table
                            , engine
                            , schema='dbo'
                            , if_exists='append'
                            , index =False
                           )

# now run the stored procedure (which will add the run_date to the volume rows as well)
exec_stored_procedure(engine, SQL_erlang_proc)

