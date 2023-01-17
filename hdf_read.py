import pandas as pd

def get_task_class_ids(hdf):
    tc_ids = []
    for row in hdf.event_names:
        if "_task" in row:
            tc_ids.append(hdf.event_types[row])
            
    return tc_ids

hdf = pd.HDFStore("test_profile_output-0.prof.h5")

print(hdf.events.head())
