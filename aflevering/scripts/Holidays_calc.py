#%%
import datetime as dt
from dateutil.easter import easter
import holidays

#%%


def get_date_type(date: dt.date) -> int:
    '''
    Function for determining a datetype. We are operating with 2 datetypes,
    0, which is a workday and 1, which is a holiday or a weekend. We consider
    the days between Christmas and New Year as holidays, as we do with the
    day after Ascension day.

    Parameters
    ----------
    date : dt.date
        The date to evaluate.

    Returns
    -------
    date_type : int
        The type of day the date is. Either 1 for normal workday, and 2
        for weekends and holidays.

    '''
    dk_holidays = holidays.CountryHoliday('DNK')
    day_after_ascention = easter(date.year) + dt.timedelta(days=40)
    special_days = []
    special_days.append(dt.date(date.year, 12, 24))
    special_days.append(dt.date(date.year, 12, 27))
    special_days.append(dt.date(date.year, 12, 28))
    special_days.append(dt.date(date.year, 12, 29))
    special_days.append(dt.date(date.year, 12, 30))
    special_days.append(dt.date(date.year, 12, 31))
    special_days.append(day_after_ascention)

    if ((date in dk_holidays) or (date.weekday() in [5, 6])
       or (date in special_days)):
        date_type = 1
    else:
        date_type = 0
    return date_type
#%%

today = dt.date(2021,1,4)
get_date_type(today)
# %%
