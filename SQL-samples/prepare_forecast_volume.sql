/*
  DESCRIPTION:

  This query is used to prepare data on call and chat volume for use in a Python program that will
  forecast them at a half-hourly basis for the following two weeks. The first four CTE tables aggregate
  call and chat volume into half-hour buckets. The main query combines the bucketed data and adds holiday
  information. This table is then passed to a python program which uses it to create a forecast.
*/


use network_services

drop table if exists #tmp

;
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
					  or (calendar_year=2020 AND is_holiday=1) -- the OR part accounts for missing federal holiday data prior to 2021
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
	into #tmp
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
where t1.weekstart < DATEADD(ww,DATEDIFF(ww,0,getdate()),-1) -- drop data from the current (partial) week
order by t1.bucket



select weekstart, interaction_type, sum(volume) as vol
from #tmp
group by weekstart, interaction_type
order by weekstart, interaction_type
