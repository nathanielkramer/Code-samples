import pandas as pd
# import pyodbc
import sqlalchemy as sqla


# m_cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
#                       "Server= cmmsqlrpt2;"
#                       "Database=network_services;"
#                       "Trusted_Connection=yes;")

# having issues with a weird error, try the SQL method I usually use
# create SQL connection to the network services db here
# create connection
server = 'CMMSQLRPT2'
db = 'network_services'
driver = 'SQL+Server'
conn_str = 'mssql+pyodbc://{}/{}?driver={}'.format(server,db,driver)
engine = sqla.create_engine(conn_str)
                      
## This query pulls all dates that are not currently in the output table from the input table.
# QUERY MODIFIED TO ADD FORECAST_WEEK and CHANGE TABLES; UPDATE IN FINAL VERSION

m_query = """select run_date, category, interaction_type, forecast_week, day_of_week, hour, average
--from network_services.dbo.t_wfm_erlang_inputs
from network_services.dbo.tmp_wfm_erlang_inputs_v2_NAK
where run_date not in (select distinct cast(run_date as datetime) 
                       --from network_services.dbo.t_wfm_erlang_outputs
                       from network_services.dbo.tmp_wfm_erlang_outputs_v2_NAK
                       )"""

m_df = pd.read_sql_query(m_query,engine)
#m_cnxn.close()


#"""
#The following code blocks are copied from pyworkforce. This is a public repo for an Erlang C model.
#https://github.com/rodrigo-arenas/pyworkforce
#I'm only brining in the parts relevant to the output of the Erlang C model so it's not quite as robust as the full package.
#If we can load the package into the system we won't need this section
#"""

from math import exp, ceil, floor
from collections.abc import Mapping, Iterable
from itertools import product
from functools import partial, reduce
import operator
import numpy as np

class ParameterGrid:
    """
    This implementation is taken from scikit-learn: https://github.com/scikit-learn/scikit-learn
    Grid of parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.
    Read more in the :ref:`User Guide <grid_search>`.
    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        An empty dict signifies default parameters.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')
        
class ErlangC:
    def __init__(self, transactions: float, aht: float, asa: float,
                 interval: int = None, shrinkage=0.0,
                 **kwargs):
        """
        Computes the number of positions required fo attend a number of transactions in a queue system based on ErlangC
        Implementation based on: https://lucidmanager.org/data-science/call-centre-workforce-planning-erlang-c-in-r/
        :param transactions: number of total transactions that comes in an interval
        :param aht: average handling time of a transaction (minutes)
        :param asa: Required average speed of answer in minutes
        :param interval: Interval length (minutes)
        :param shrinkage: Percentage of time that an operator unit is not available
        """

        if transactions <= 0:
            raise ValueError("transactions can't be smaller or equals than 0")

        if aht <= 0:
            raise ValueError("aht can't be smaller or equals than 0")

        if asa <= 0:
            raise ValueError("asa can't be smaller or equals than 0")

        if interval <= 0:
            raise ValueError("interval can't be smaller or equals than 0")

        if shrinkage < 0 or shrinkage >= 1:
            raise ValueError("shrinkage must be between in the interval [0,1)")

        self.n_transactions = transactions
        self.aht = aht
        self.interval = interval
        self.asa = asa
        self.intensity = (self.n_transactions / self.interval) * self.aht
        self.shrinkage = shrinkage

    def waiting_probability(self, positions, scale_positions=False):
        """
        :param positions: Number of positions to attend the transactions
        :param scale_positions: True if the positions where calculated using shrinkage
        :return: the probability of a transaction waits in queue
        """
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        erlang_b_inverse = 1
        for position in range(1, productive_positions + 1):
            erlang_b_inverse = 1 + (erlang_b_inverse * position / self.intensity)

        erlang_b = 1 / erlang_b_inverse
        return productive_positions * erlang_b / (productive_positions - self.intensity * (1 - erlang_b))

    def service_level(self, positions, scale_positions=False):
        """
        :param positions: Number of positions attending
        :param scale_positions: True if the positions where calculated using shrinkage
        :return: achieved service level
        """
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        probability_wait = self.waiting_probability(productive_positions, scale_positions=False)
        exponential = exp(-(productive_positions - self.intensity) * (self.asa / self.aht))
        return max(0, 1 - (probability_wait * exponential))

    def achieved_occupancy(self, positions, scale_positions=False):
        """
        :param positions: Number of raw positions
        :param scale_positions: True if the positions where calculated using shrinkage
        :return: Expected occupancy of positions
        """
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        return self.intensity / productive_positions

    def required_positions(self, service_level: float, max_occupancy: float = 1.0):
        """
        :param service_level: Target service level
        :param max_occupancy: Maximum fraction of time that an attending position can be occupied
        :return:
                 * raw_positions: Required positions assuming shrinkage = 0
                 * positions: Number of positions needed to ensure the required service level
                 * service_level: Fraction of transactions that are expect to be assigned to a position,
                   before the asa time
                 * occupancy: Expected occupancy of positions
                 * waiting_probability: The probability of a transaction waits in queue
        """

        if service_level < 0 or service_level > 1:
            raise ValueError("service_level must be between 0 and 1")

        if max_occupancy < 0 or max_occupancy > 1:
            raise ValueError("max_occupancy must be between 0 and 1")

        positions = round(self.intensity + 1)
        achieved_service_level = self.service_level(positions, scale_positions=False)
        while achieved_service_level < service_level:
            positions += 1
            achieved_service_level = self.service_level(positions, scale_positions=False)

        achieved_occupancy = self.achieved_occupancy(positions, scale_positions=False)

        raw_positions = ceil(positions)

        if achieved_occupancy > max_occupancy:
            raw_positions = ceil(self.intensity / max_occupancy)
            achieved_occupancy = self.achieved_occupancy(raw_positions)
            achieved_service_level = self.service_level(raw_positions)

        waiting_probability = self.waiting_probability(positions=raw_positions)
        positions = ceil(raw_positions / (1 - self.shrinkage))

        return {"raw_positions": raw_positions,
                "positions": positions,
                "service_level": achieved_service_level,
                "occupancy": achieved_occupancy,
                "waiting_probability": waiting_probability}
    
#"""
#This is the end of the copied code
#"""

## Testing an example of the ErlancC class
## This code allows manual calculation for any selected inputs

#erlang = ErlangC(transactions=14.6, asa=20/60, aht=272/60, interval=30, shrinkage=0.4)

#positions_requirements = erlang.required_positions(service_level=0.85, max_occupancy=0.99)
#print("positions_requirements: ", positions_requirements)


## Build Call Volume, HandleTime, Occupancy, and Shrinkage dataframes
#NAK MODIFIED: modified drop steps below to drop forecast_week where needed
call_vol = m_df[(m_df["category"] == "Volume") & (m_df["interaction_type"] == "Call")]
call_vol = call_vol.rename(columns={"average": "avg_vol"}).drop(columns=["category"])

call_ht = m_df[(m_df["category"] == "HandleTime") & (m_df["interaction_type"] == "Call")]
call_ht = call_ht.rename(columns={"average": "avg_ht_sec"}).drop(columns=["category", "interaction_type", "forecast_week", "hour"])

call_occ = m_df[(m_df["category"] == "OccupancyRate") & (m_df["interaction_type"] == "Call")]
call_occ = call_occ.rename(columns={"average": "input_occupancy"}).drop(columns=["category", "interaction_type", "forecast_week"]).round({'input_occupancy':2})

call_srk = m_df[(m_df["category"] == "Shrinkage")]
call_srk = call_srk.rename(columns={"average": "shrinkage"}).drop(columns=["category", "interaction_type", "forecast_week","day_of_week", "hour"])


## Using our Call tables, we join them together and add the columns we need for the final insert
call_df = pd.merge(call_vol, call_ht, on=["run_date", "day_of_week"], how="inner")
call_df = pd.merge(call_df, call_occ, on=["run_date", "day_of_week", "hour"], how="inner")
call_df = pd.merge(call_df, call_srk, on=["run_date"], how="inner")
call_df = call_df.reindex(columns = [*call_df.columns.tolist(), "raw_positions", "positions", "service_level", "occupancy", "waiting_probability"], fill_value = 0.0)

## This loop will apply the Erlang C model to every row in the dataframe.
## Current hard coded values
##     Average Speed of Answer 20/60 = 20 seconds
##     Interval 30 = 30 minutes
##     Service Level 0.85 = 85%

for index, row in call_df.iterrows():
    erlang = ErlangC(transactions=row["avg_vol"], asa=23/60, aht=row["avg_ht_sec"]/60, interval=30, shrinkage=row["shrinkage"])
    positions_requirements = erlang.required_positions(service_level=0.95, max_occupancy=max(0.75,row["input_occupancy"]))
    
    call_df.at[index,"raw_positions"] = positions_requirements["raw_positions"]
    call_df.at[index,"positions"] = positions_requirements["positions"]
    call_df.at[index,"service_level"] = positions_requirements["service_level"]
    call_df.at[index,"occupancy"] = positions_requirements["occupancy"]
    call_df.at[index,"waiting_probability"] = positions_requirements["waiting_probability"]


## Now that Calls have been set up we output

# engine = sqla.create_engine('mssql+pyodbc://cmmsqlrpt2.innova.local/network_services?driver=SQL Server Native Client 11.0?Trusted_connection-yes', poolclass=sqla.pool.NullPool)
call_df.to_sql('tmp_wfm_erlang_outputs_v2_NAK', con = engine, schema='dbo', if_exists='append') 


## Repeat for Chats
## Build Chat Volume, HandleTime, Concurrency, Occupancy, and Shrinkage dataframes
#NAK MODIFIED: modified drop steps below to drop forecast_week where needed
chat_vol = m_df[(m_df["category"] == "Volume") & (m_df["interaction_type"] == "Chat")]
chat_vol = chat_vol.rename(columns={"average": "avg_vol"}).drop(columns=["category"])

chat_ht = m_df[(m_df["category"] == "HandleTime") & (m_df["interaction_type"] == "Chat")]
chat_ht = chat_ht.rename(columns={"average": "avg_ht_sec"}).drop(columns=["category", "interaction_type", "forecast_week", "hour"])

chat_con = m_df[(m_df["category"] == "Concurrency") & (m_df["interaction_type"] == "Chat")]
chat_con = chat_con.rename(columns={"average": "chat_concurrency"}).drop(columns=["category", "interaction_type", "forecast_week"])

## Use Concurrency to reduce chat handle time
chat_ht = pd.merge(chat_ht, chat_con, on=["run_date", "day_of_week"], how="inner")
chat_ht["avg_ht_sec"] = chat_ht["avg_ht_sec"] / chat_ht["chat_concurrency"]
chat_ht = chat_ht.drop(columns=["chat_concurrency"]).round({'avg_ht_sec':2})

## Shrinkage is the same for both interaction types, we'll just use the prebuilt call_srk table


chat_df = pd.merge(chat_vol, chat_ht, on=["run_date", "day_of_week", "hour"], how="inner")
chat_df = pd.merge(chat_df, call_srk, on=["run_date"], how="inner")
chat_df = chat_df.reindex(columns = [*chat_df.columns.tolist(), "raw_positions", "positions", "service_level", "occupancy", "waiting_probability"], fill_value = 0.0)


## This loop will apply the Erlang C model to every row in the dataframe.
## Current hard coded values
##     Average Speed of Answer 20/60 = 20 seconds
##     Interval 30 = 30 minutes
##     Service Level 0.85 = 85%
##     Occupancy 0.75 = 75%

for index, row in chat_df.iterrows():
    erlang = ErlangC(transactions=row["avg_vol"], asa=20/60, aht=row["avg_ht_sec"]/60, interval=30, shrinkage=row["shrinkage"])
    positions_requirements = erlang.required_positions(service_level=0.85, max_occupancy=0.75)
    
    chat_df.at[index,"raw_positions"] = positions_requirements["raw_positions"]
    chat_df.at[index,"positions"] = positions_requirements["positions"]
    chat_df.at[index,"service_level"] = positions_requirements["service_level"]
    chat_df.at[index,"occupancy"] = positions_requirements["occupancy"]
    chat_df.at[index,"waiting_probability"] = positions_requirements["waiting_probability"]

## Now that Chats have been set up we output

# engine = sqla.create_engine('mssql+pyodbc://cmmsqlrpt2.innova.local/network_services?driver=SQL Server Native Client 11.0?Trusted_connection-yes', poolclass=sqla.pool.NullPool)
chat_df.to_sql('tmp_wfm_erlang_outputs_v2_NAK', con = engine, schema='dbo', if_exists='append') 
