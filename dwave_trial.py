from dimod import Integer, Binary
from dimod import quicksum 
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler
from dwave.optimization.mathematical import add, logical_or, maximum, minimum
from dwave.optimization.model import Model

import pandas as pd
from datetime import (
    datetime,
    time as Time
)
from numpy import (
    nan
)
import time
from constants import (
    LaborGenColumns as L, 
    AssociateColumns as A,
    ModelColumns as M,
    ExtraColumns as E
)

HOURS_PER_DAY = 24
PERIODS_PER_HR = {
    'QH': 4, # Quarter-Hr
    'HH': 2, # Half-Hr
    'HO': 1  # Hourly
}
HORIZON = {
    'QH': PERIODS_PER_HR['QH'] * HOURS_PER_DAY, 
    'HH': PERIODS_PER_HR['HH'] * HOURS_PER_DAY,
    'HO': PERIODS_PER_HR['HO'] * HOURS_PER_DAY,
}
GRANULARITY = {'QH': 15, 'HH': 30, 'HO': 60}
FT = 'FT'
PT = 'PT'
JOB_ROLES_MAPPING = {
    'Asst. Bakery Manager', 
    'Baker', 
    'Bakery Apprentice',
    'Bakery Clerk', 
    'Bakery Manager', 
    'Decorator',
    'Asst. Store Manager', 
    'Beverage Clerk', 
    'Cashier',
    'Cust Serv Team Leader', 
    'Custodian', 
    'Customer Service Manager',
    'Customer Service Staff', 
    'Front Service Clerk',
    'Lead Beverage Clerk', 
    'Store Manager',
    'Floating Asst Bakery Manager', 
    'Asst. Cust Serv Mgr'
}

def create_data_model(laborgen_path, associate_path, store_id: int = 10, department: str = 'Bakery') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates the dataframes that describes the model."""
    laborgen = pd.read_csv(laborgen_path)
    assoc = pd.read_csv(associate_path)
    assoc = assoc.loc[lambda f: (f[A.store] == store_id)&(f[A.dept_nm] == department)] #Chose random store which may be different from laborgen

    return laborgen, assoc

def str_to_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col])

def filter_labor_granularity(labor: pd.DataFrame, granularity: str = 'QH') -> pd.DataFrame:
    # Filter rows based on granularity
    if granularity in ['HH' , 'HO']:
        labor = labor.loc[
                lambda f: f[L.start_time].dt.minute % GRANULARITY[granularity] == 0
        ].reset_index(drop=True)
    return labor

def period_to_time(period: int, granularity: str = 'QH'):
    hour = int(period / PERIODS_PER_HR[granularity])
    minute = GRANULARITY[granularity] * (period % PERIODS_PER_HR[granularity])
    return Time(hour, minute)

def dayofweek_to_num(df: pd.DataFrame, col: str):
    # Converting day of week to num. Sunday=0, Monday=1, etc
    return df[col].apply(lambda x: (x.dayofweek+1) % 7)

def time_to_period(df: pd.DataFrame, col: str, granularity: str = 'QH'):
    # Converting time of day to periods
    return df[col].apply(
        lambda x: int(PERIODS_PER_HR[granularity]*x.hour + int(x.minute/GRANULARITY[granularity]))
    )

def convert_to_period(timestamp: pd.Timestamp, granularity: str = 'QH') -> int:
    return int(
                PERIODS_PER_HR[granularity]*timestamp.hour + \
                timestamp.minute/GRANULARITY[granularity]
            )

def is_FTE(df: pd.DataFrame, col: str):
    return df[col].apply(lambda x: PT not in x.split(' '))

def extract_labor_demand(df: pd.DataFrame, closing_period: int, granularity: str = 'QH'):
    """
        Extracts 2 lists to create labor demand step function. df should only have
        one unique role
    """
    df = df.sort_values(L.period).reset_index(drop=True)
    indices = [0]
    ref = [0]
    for period, labor_val in df[[L.period, L.labor_val]].values:
        period = int(period)
        labor_val = int(labor_val)
        if labor_val != ref[-1]:
            indices.append(period)
            ref.append(labor_val)
    if period == closing_period:
        indices.append(period)
    else:
        indices.append(period+1) #assumption that next period 
    ref.append(0)

    # Need to include the endpoint just incase it wasn't included in the above for-loop. indices should have one more than ref
    indices.append(HOURS_PER_DAY * PERIODS_PER_HR[granularity])
    indices = sorted(list(set(indices)))
    return indices, ref

def create_main_vars(model: Model, associate: pd.DataFrame, labor: pd.DataFrame, 
        days_of_week: list[int] = [0], granularity: str = 'QH') -> list[pd.DataFrame]:
    """
        Dataframe with multi-Index (day, employee, role) and following columns
            - day_num
            - employee_id
            - role
            - isFT
            - start: variable
            - end: variable
            - performed: column which contains bool variable
            - interval: column which contains interval variable

        Returns:
            - main variable df at the most granular level
            - variable df at the daily level
    """
    x = []
    emp_data_dic = associate.set_index(A.employee_id).to_dict('index')
    num_unique_roles = labor[L.role_id].nunique()
    roles_order = labor[L.role_id].unique()
    lb = labor[L.period].min()
    ub = labor[L.period].max()
    for (day, emp), df in labor.groupby([L.day_num, A.employee_id]):
        start = model.integer(
            num_unique_roles, 
            lower_bound=lb, 
            upper_bound=ub
        )
        dur = model.integer(
            num_unique_roles, 
            lower_bound=0, 
            upper_bound=PERIODS_PER_HR[granularity]*10
        )

        # creating shift vars
        shift_start = minimum(start) #TODO
        shift_end = maximum(start + dur) #TODO
        x.append([day, emp, start, dur, shift_start, shift_end])
    
    cols = [
        L.day_num, 
        A.employee_id, 
        M.role_start, 
        M.role_duration, 
        M.shiftStart, 
        M.shiftEnd
    ]
    vars = pd.DataFrame(data=x, columns=cols)

    return vars, roles_order

def zero_out_role_vars(model: Model, vars: pd.DataFrame, associate: pd.DataFrame, job_role: pd.DataFrame, ordered_roles: list):
    vars_dict = vars.set_index([L.day_num, A.employee_id]).to_dict('index')
    
    # Will zero out variables for roles which an employee cannot perform
    for emp, job_name in associate[[A.employee_id, A.primary_job]].drop_duplicates().values:
        df_job_role = job_role.loc[lambda f: f[A.job_name]==job_name]
        for role in ordered_roles:
            if role not in df_job_role[L.role_id].to_list():
                empDaySet = [vars_dict[day, emp] for day in range(7)] 
                # zeroing shift vars employees cannot work
                for v in empDaySet:
                    model.Add(v[M.role_start]==0)
                    model.Add(v[M.role_duration]==0)

def add_cumulative_constraints(
        model: Model, labor: pd.DataFrame, associate: pd.DataFrame, 
        variables: pd.DataFrame, days_of_week: list[int] = [0], granularity: str = 'QH'):
    """
        Creates cumulative constraint which enforces that at each continuous time 't'
        demand is met, ie if 2 associates are needed at 12:30PM, then we have at least 
        that many. Therefore, we create this constraint for each day in days_of_week 
        (which would be Sunday-Saturday in production) and role.
    """
    for 

def add_role_continuity(model: Model, vars: pd.DataFrame, ordered_roles: list, associate: pd.DataFrame, job_role: pd.DataFrame, continuity: int, granularity: str = 'QH'):
    vars_dict = vars.set_index([L.day_num, A.employee_id]).to_dict('index')

    for emp, job_name in associate[[A.employee_id, A.primary_job]].drop_duplicates().values:
        df_job_role = job_role.loc[lambda f: f[A.job_name]==job_name]
        for role in ordered_roles:
            if role in df_job_role[L.role_id].to_list():
                for day in vars[L.day_num].unique():
                    model.Add(vars_dict[day, emp][M.role_duration] >= continuity*PERIODS_PER_HR[granularity])

def add_no_overlap_constraints(model: Model, variables: pd.DataFrame, 
                               daily_variables: pd.DataFrame, days_of_week: list[int] = [0]):
    pass

def add_min_hours_between_shift(
        model: Model, variables: pd.DataFrame, associate: pd.DataFrame, 
        min_hours_between_shift: int = 10, days_of_week: list[int] = [0],
        granularity: str = 'QH'):
        pass

def add_weekly_max_work_days(model: Model, variables: pd.DataFrame, weekly_max_work_days: int = 5):
    pass
            
def add_weekly_hours(model: Model, variables: pd.DataFrame, associate: pd.DataFrame, granularity: str = 'QH'):
    pass

def add_min_role_worktime(model: Model, variables: pd.DataFrame, min_role_hrs: int, 
                        days_of_week: list[int] = [0], granularity: str = 'QH'):
    pass

def add_role_rank(model: Model, vars: pd.DataFrame, job_role: pd.DataFrame, associate: pd.DataFrame, roles_order: list):
    cols = [L.role_id, 'num', M.slack]
    slack_list = []
 
    vars_dict = vars.set_index([L.day_num, A.employee_id]).to_dict('index')

    job_role = job_role.loc[
        lambda f: f[L.role_id].isin(roles_order),
        [L.role_id, A.rank_level, A.job_name]
    ]
    associate = associate[[A.employee_id, A.primary_job]]
    vars = vars.merge(
            associate,
            on=A.employee_id,
            how='left'
    )
    vars = vars.merge(
            job_role,
            how='left',
            left_on=[A.primary_job],
            right_on=[A.job_name]
    ).fillna({A.rank_level: 30})

    for role in roles_order:
        # sorted since small number means higher rank
        kys = sorted(list(vars.loc[lambda f: f[L.role_id]==role, A.rank_level].unique()))
        for k1, k2 in zip(kys, kys[1:]):
            empSet1 = vars.loc[lambda f: f[A.rank_level] == k1, A.employee_id]
            empSet2 = vars.loc[lambda f: f[A.rank_level] == k2, A.employee_id]
            sumVar1 = [
                vars_dict[daynum, emp][] 
                    for daynum in range(7) 
                    for emp in empSet1
            ]
            sumVar2 = [
                vars_dict[daynum, emp][] 
                    for daynum in range(7) 
                    for emp in empSet2
            ]
            int_var = model.integer(
                lb=0, ub = 100*HOURS_PER_DAY
            )
            slack_list.append([role, k1, int_var])
            model.Add(var1.sum() >= var2.sum())
       
    slacks_df = pd.DataFrame(data=slack_list, columns=cols)
    return slacks_df

def add_objective(model: Model, variables: pd.DataFrame):
    pass

def main(laborgen_path: str, associate_path: str, store_id: int, 
         department: str, graph_soln: bool = False, granularity: str = 'QH') -> None:
    
    # Setting timer to measure runtime
    start_time = time.time()

    """
        Data Prep
    """
    print('Preparing data ...')
    laborgen, assoc = create_data_model(
        laborgen_path=laborgen_path, 
        associate_path=associate_path,
        store_id=store_id,
        department=department
    )
    str_to_timestamp(laborgen, L.start_time)
    laborgen = filter_labor_granularity(laborgen, granularity=granularity)
    laborgen[L.day_num] = dayofweek_to_num(laborgen, L.start_time)
    laborgen[L.period] = time_to_period(laborgen, L.start_time, granularity=granularity)
    assoc[A.isFT] = is_FTE(assoc, A.shift_strat)
    start_date = laborgen[L.start_time].min()
    
    """
        Dwave modeling begins with variable creation and next adding constraints
    """
    # Create the model.
    

    print(f"E2E took {round((time.time()-start_time)/60, 3)} minutes.")

if __name__ == '__main__':
    department = 'Customer Service'
    if department == 'Customer Service':
        laborgen_filename = 'demand'
        laborgen_path = f'/workspaces/dwave-trial/data/{laborgen_filename}.csv'

    associate_filename = 'associates'
    associate_path = f'/workspaces/dwave-trial/data/{associate_filename}.csv'

    main(
        laborgen_path, 
        associate_path, 
        store_id=10,
        department=department,
        graph_soln = True,
        granularity='QH'
    )


