import datetime


def get_current_time_in_string():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
