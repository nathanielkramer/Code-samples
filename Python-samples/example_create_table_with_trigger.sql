-- =============================================
-- Author:			Nathaniel Kramer

-- Create date:		2022-08-10  

-- Description:		This is a script that I wrote to create a table in my team's SQL Server database to automatically
--                update our forecast program records based on the creation of a new forebast by our Finance team.
--                It uses a trigger to run UPDATE and INSERT statements each time a new Archive_ID is created in the
--                Finance table (which is the signal that a new forecast has been created).
--                

--          Create table for storing historical outputs of the network services forecast program. This program
--					exists in Python and is an updated version of the previous Excel-based forecast. It takes the finance
--					forecast (stored in network_services..t_wfm_forecast_volume) and converts it to a headcount need forecast.

--					The second part of this code creates a trigger on the table which will auto-populate the history table 
--					each time the forecast program is run. When the forecast program runs, it outputs to the table referenced
--					in the trigger, network_services..tmp_STAGING_wfm_forecast_output.

-- Updates: 


-- =============================================

USE [network_services]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- create the actual table for the forecast history data
CREATE TABLE [dbo].[t_wfm_NS_forecast_output] (
	  ID int IDENTITY(1,1) PRIMARY KEY
	, NS_Run_Date datetime
	, Forecast_Month datetime
	, archive_id int
	, Volume float
	, AC_need float
	, AC_CMM_Attrition float
	, AC_Fortuity float
	, AC_Fortuity_Attrition float
	, AC_Senior_Role float
	, AS_need float
	, AS_CMM_Attrition float
	, AS_Fortuity float
	, AS_Fortuity_Attrition float
	, created_date datetime DEFAULT GETDATE()
	, modified_date datetime DEFAULT NULL
	);

GO

------------------------------------------------------------
-- Code to create trigger

CREATE TRIGGER [dbo].[NS_forecast_update_trigger]
ON [dbo].[tmp_STAGING_wfm_forecast_output]
	AFTER INSERT, UPDATE
	AS
	BEGIN
	SET NOCOUNT ON;


-- create the table structure that we will use  by renaming and reordering some of the columns from the Python output table
drop table if exists #forecast_output
select 
	NS_Run_Date
	, Month as Forecast_Month
	, CAST(archive_id as int) archive_id --stored as bigint in the python transfer
	, Volume
	, [AC EOM Need] as AC_need
	, [AC CMM Attrition] as AC_CMM_Attrition
	, [AC Fortuity] as AC_Fortuity
	, [AC Fortuity Attrition] as AC_Fortuity_Attrition
	, [AC Senior Role] as AC_Senior_Role
	, [AS EOM Need] as AS_need
	, [AS CMM Attrition] as AS_CMM_Attrition
	, [AS Fortuity] as AS_Fortuity
	, [AS Fortuity Attrition] as AS_Fortuity_Attrition
into #forecast_output
from network_services.dbo.tmp_STAGING_wfm_forecast_output



----------------------------------------------------------------------------------------------------------
--		CODE FOR UPDATING HISTORY TABLE
--		Insert new lines for every new archive_id, but UPDATE old data if there are changes
----------------------------------------------------------------------------------------------------------


-- check for any rows with data that don't agree with existing rows
drop table if exists #stage_updates
select	 t2.NS_Run_Date
		,t2.Forecast_Month
		,t2.archive_id
		,t2.Volume
		,t2.AC_need
		,t2.AC_CMM_Attrition
		,t2.AC_Fortuity
		,t2.AC_Fortuity_Attrition
		,t2.AC_Senior_Role
		,t2.AS_need
		,t2.AS_CMM_Attrition
		,t2.AS_Fortuity
		,t2.AS_Fortuity_Attrition
into #stage_updates
from network_services.dbo.t_wfm_NS_forecast_output t1
left join #forecast_output t2 ON
	t1.Forecast_Month = t2.Forecast_Month
	AND t1.archive_id = t2.archive_id
WHERE  t1.Volume != t2.Volume
	OR t1.AC_need != t2.AC_need
	OR t1.AC_CMM_Attrition != t2.AC_CMM_Attrition
	OR t1.AC_Fortuity != t2.AC_Fortuity
	OR t1.AC_Fortuity_Attrition != t2.AC_Fortuity_Attrition
	OR t1.AC_Senior_Role != t2.AC_Senior_Role
	OR t1.AS_need != t2.AS_need
	OR t1.AS_CMM_Attrition != t2.AS_CMM_Attrition
	OR t1.AS_Fortuity != t2.AS_Fortuity
	OR t1.AS_Fortuity_Attrition != t2.AS_Fortuity_Attrition


-- actually do the updates
UPDATE h
SET  h.NS_Run_Date = u.NS_Run_Date
	,h.Volume = u.Volume
	,h.AC_need = u.AC_need
	,h.AC_CMM_Attrition = u.AC_CMM_Attrition
	,h.AC_Fortuity = u.AC_Fortuity
	,h.AC_Fortuity_Attrition = u.AC_Fortuity_Attrition
	,h.AC_Senior_Role = u.AC_Senior_Role
	,h.AS_need = u.AS_need
	,h.AS_CMM_Attrition = u.AS_CMM_Attrition
	,h.AS_Fortuity = u.AS_Fortuity
	,h.AS_Fortuity_Attrition = u.AS_Fortuity_Attrition
	,h.modified_date = getdate() --update the modified column
FROM network_services.dbo.t_wfm_NS_forecast_output h
	join #stage_updates u
ON 	h.Forecast_Month = u.Forecast_Month
	AND h.archive_id = u.archive_id


-- Insert the new rows into the data. Note that these are just any row with an Archive ID not already in the table, which SHOULD be sufficient to give us our desired output

BEGIN
	INSERT INTO network_services.dbo.t_wfm_NS_forecast_output
		( NS_Run_Date
		, Forecast_Month
		, archive_id 
		, Volume
		, AC_need
		, AC_CMM_Attrition
		, AC_Fortuity
		, AC_Fortuity_Attrition
		, AC_Senior_Role
		, AS_need
		, AS_CMM_Attrition
		, AS_Fortuity
		, AS_Fortuity_Attrition
		, created_date
		, modified_date)

	select 
		NS_Run_Date
		, Forecast_Month
		, archive_id 
		, Volume
		, AC_need
		, AC_CMM_Attrition
		, AC_Fortuity
		, AC_Fortuity_Attrition
		, AC_Senior_Role
		, AS_need
		, AS_CMM_Attrition
		, AS_Fortuity
		, AS_Fortuity_Attrition
		, getdate() -- created date
		, CAST(NULL as datetime) --modified date
	from #forecast_output
	where archive_id not in (select distinct archive_id from network_services.dbo.t_wfm_NS_forecast_output)

END


END;
