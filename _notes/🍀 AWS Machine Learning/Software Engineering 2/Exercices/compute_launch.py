
def days_until_launch(current_day, launch_day):
    """"Returns the days left before launch.
    
    current_day (int) - current day in integer
    launch_day (int) - launch day in integer
    """
    if current_day >= launch_day:
        dul = 0
    else:
        dul = launch_day - current_day
    return dul 
