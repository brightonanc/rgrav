# utilities for benchmarking/timing algorithms

import time

def time_func(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end - start, result

def average_time(func, n_trials, *args, **kwargs):
    times = []
    for i in range(n_trials):
        times.append(time_func(func, *args, **kwargs))
    return sum(times) / n_trials

class TimeAccumulator:
    def __init__(self):
        self.times = []

    def time_func(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        self.times.append(end - start)
        return result

    def add_time(self, time):
        self.times.append(time)

    def total_time(self):
        return sum(self.times)

    def mean_time(self):
        return sum(self.times) / len(self.times)
