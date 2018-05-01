'''
 ' Given start and stop points and a time period, get a linear
 ' annealing rate.
'''
def get_annealing_rate(start, stop, time):
  return (stop - start) / time

'''
 ' Given an annealing rate, a starting point, and a time, get a leanearly
 ' annealed value.
'''
def get_annealed_value(rate, start, time):
  return rate * time + start

