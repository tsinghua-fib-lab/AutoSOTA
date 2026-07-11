import time

def time_one():
    start = time.time()  # 1657267196.3012242

    end = time.time()  # 1657267201.6171696
    return end - start

# start =time.clock()  # 3.3892019

# end = time.clock()  # 8.7225603
# print('spend： %s second' % (end - start))

# start = timeit.default_timer()  # start： 3.3247171

# end = timeit.default_timer()  # end： 8.5791582
# print('spend： %s second' % (end - start))
