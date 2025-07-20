from dataclasses import dataclass

@dataclass
class LaborGenColumns():
    role_id: str = 'RoleId'
    store: str = 'Store'
    dept_id: str = 'DepartmentId'
    business_date: str = 'BusinessDate'
    start_time: str = 'StartTime'
    day_num: str = 'day_num'
    period: str = 'period'
    labor_val: str = 'LaborValue'
    role_name: str = 'RoleName'
    time: str = 'time'
    week: str = 'week'

@dataclass
class ModelColumns():
    start: str = 'start'
    duration: str = 'duration'
    end: str = 'end'
    performed: str = 'performed'
    interval: str = 'interval'
    break_interval: str = 'break_interval'
    break_start: str = 'break_start'
    break_end: str = 'break_end'
    bool_var: str = 'bool_var'
    int_var: str = 'int_var'
    solution: str = 'solution'
    bool_xps: str = 'bool_xps'

@dataclass
class ExtraColumns():
    date: str = 'date'
    leftover: str = 'leftover'
    overcoverage: str = 'overcoverage'
    res: str = 'res'
    breakr: str = 'break'
    break_roleID: int = 9999999
    unavailable: str = 'unavailable'
    rank: str= 'rank'
    extended_duration: str = 'extended_duration'
    extended_start: str = 'extended_start'
    temp: str = 'temp'
    end_time: str = 'end_time'
    all_roles: str = 'all_roles'
    morning: str = 'morning'
    evening: str = 'evening'
    is_working: str = 'is_working'
    max_hrs: str = 'max_hours'
    earliest: str = 'earliest'
    latest: str = 'latest'
    meal: str = 'meal'
    pos: str = 'pos'

@dataclass
class AvailabilityColumns():
    start_date: str = 'start_date'
    end_date: str = 'end_date'
    avail_all_day: str = 'is_all_day'
    day_id: str = 'day_of_week_id'
    start_time: str = 'start_time_offset'
    end_time: str = 'end_time_offset'

@dataclass
class AssociateColumns():
    store: str = 'Home_Store'
    employee_id: str = 'employee_id'
    first_nm: str = 'first_name'
    last_nm: str = 'last_name'
    dept_nm: str = 'Department'
    primary_job: str = 'Primary_Job'
    job_name: str = 'JobName'
    jobID: str = 'JobId'
    rank_level: str = 'rank_level'
    shift_strat: str = 'Shift_Strategy'
    shift_strat_rank: str = 'shift_strategy_rank'
    weekly_min_hrs: str = 'weekly_hours_min'
    weekly_max_hrs: str = 'weekly_hours_max'
    max_consecutive_days: str = 'max_consecutive_days'
    min_hours_between_shifts: str = 'min_hours_between_shifts'
    seq_num: str = 'SequenceNumber'
    strict_weekly_min_hrs: str = 'strict_weekly_hours_min'
    isFT: str = 'isFT'
    seniority_date: str = 'seniority_date'
    emp_age: str = 'age'
    birthdate: str = 'birthdate'