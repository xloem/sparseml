import datetime

import matplotlib.pyplot as plt

log = """
TIME:07:39:24 7328
TIME:07:39:26 7424
TIME:07:39:28 7424
TIME:07:39:30 7424
TIME:07:39:32 7712
TIME:07:39:34 7808
TIME:07:39:36 7904
TIME:07:39:38 7904
TIME:07:39:40 8192
TIME:07:39:42 8192
TIME:07:39:44 8192
TIME:07:39:46 8192
TIME:07:39:48 8192
TIME:07:39:50 8192
TIME:07:39:52 8192
TIME:07:39:54 8192
TIME:07:39:56 8288
TIME:07:39:58 8576
TIME:07:40:00 8864
TIME:07:40:02 8864
TIME:07:40:04 8864
TIME:07:40:06 8288
TIME:07:40:08 8288
TIME:07:40:10 8384
TIME:07:40:12 8384
TIME:07:40:14 8384
TIME:07:40:16 8384
TIME:07:40:18 8384
TIME:07:40:20 8384
TIME:07:40:22 8384
TIME:07:40:24 8576
TIME:07:40:26 8576
TIME:07:40:28 7616
TIME:07:40:30 7328
TIME:07:40:32 7328
TIME:07:40:34 7328
TIME:07:40:36 7328
TIME:07:40:38 7328
"""

no_files_list = time_list = []
for entry in log.split("\n"):
    if entry:
        no_files = entry.split(" ")[-1]
        time = entry.split(" ")[0][5:]
        no_files_list.append(int(no_files))
        time_list.append(datetime.datetime.strptime(time, "%H:%M:%S"))

baseline = no_files_list[0]
old_limit = 1024 + no_files_list[0]
new_limit = 1024 * 4 + no_files_list[0]
plt.axhline(y=baseline, color='r', linestyle='-')
plt.axhline(y=old_limit, color='g', linestyle='-')
plt.axhline(y=new_limit, color='b', linestyle='-')
plt.plot(time_list, no_files_list)
plt.show()
