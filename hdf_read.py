import pandas as pd

def get_task_class_ids(hdf):
    tc_ids = []
    for row in hdf.event_names:
        if "_task" in row:
            tc_ids.append(hdf.event_types[row])

    return tc_ids

hdf = pd.HDFStore("test_profile_output-0.prof.h5")
task_class_ids = get_task_class_ids(hdf)
total_time = 0

total_tasks = 0
max_time = 0
num_tasks = 0
min_time = 99999999
task_times = []

for tc_id in task_class_ids:
    t = hdf.events[(hdf.events['tcid'] == tc_id)]

    num_tasks += len(t)
    for index, row in t.iterrows():
        time_ns = row['end'] - row['begin']
        task_times.append(time_ns / 1e3)
        max_time = max(time_ns, max_time)
        min_time = min(time_ns, min_time)
        total_time += time_ns

max_time = max_time / 1e3
min_time = min_time / 1e3
total_time = total_time / 1e3

print("HDF5:", num_tasks, ",", max_time, ",", min_time, ",",total_time)
