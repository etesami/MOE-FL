#!/usr/bin/python

DIR = "data_tmp/"

FILES = {}
FILES['20_avg'] = []
FILES['40_avg'] = []
FILES['50_avg'] = []
FILES['60_avg'] = []
FILES['80_avg'] = []
FILES['20_opt'] = []
FILES['40_opt'] = []
FILES['50_opt'] = []
FILES['60_opt'] = []
FILES['80_opt'] = []

FILES['20_avg'].append("04_attk2_avg20_20_test")
FILES['20_avg'].append("04_attk2_avg20_40_test")
FILES['20_avg'].append("04_attk2_avg20_60_test")
FILES['20_avg'].append("04_attk2_avg20_80_test")

FILES['40_avg'].append("04_attk2_avg40_20_test")
FILES['40_avg'].append("04_attk2_avg40_40_test")
FILES['40_avg'].append("04_attk2_avg40_60_test")
FILES['40_avg'].append("04_attk2_avg40_80_test")

FILES['50_avg'].append("04_attk2_avg50_20_test")
FILES['50_avg'].append("04_attk2_avg50_40_test")
FILES['50_avg'].append("04_attk2_avg50_60_test")
FILES['50_avg'].append("04_attk2_avg50_80_test")

FILES['60_avg'].append("04_attk2_avg60_20_test")
FILES['60_avg'].append("04_attk2_avg60_40_test")
FILES['60_avg'].append("04_attk2_avg60_60_test")
FILES['60_avg'].append("04_attk2_avg60_80_test")

FILES['80_avg'].append("04_attk2_avg80_20_test")
FILES['80_avg'].append("04_attk2_avg80_40_test")
FILES['80_avg'].append("04_attk2_avg80_60_test")
FILES['80_avg'].append("04_attk2_avg80_80_test")

FILES['20_opt'].append("05_attk2_opt20_20_test")
FILES['20_opt'].append("05_attk2_opt20_40_test")
FILES['20_opt'].append("05_attk2_opt20_60_test")
FILES['20_opt'].append("05_attk2_opt20_80_test")

FILES['40_opt'].append("05_attk2_opt40_20_test")
FILES['40_opt'].append("05_attk2_opt40_40_test")
FILES['40_opt'].append("05_attk2_opt40_60_test")
FILES['40_opt'].append("05_attk2_opt40_80_test")

FILES['50_opt'].append("05_attk2_opt50_20_test")
FILES['50_opt'].append("05_attk2_opt50_40_test")
FILES['50_opt'].append("05_attk2_opt50_60_test")
FILES['50_opt'].append("05_attk2_opt50_80_test")

FILES['60_opt'].append("05_attk2_opt60_20_test")
FILES['60_opt'].append("05_attk2_opt60_40_test")
FILES['60_opt'].append("05_attk2_opt60_60_test")
FILES['60_opt'].append("05_attk2_opt60_80_test")

FILES['80_opt'].append("05_attk2_opt80_20_test")
FILES['80_opt'].append("05_attk2_opt80_40_test")
FILES['80_opt'].append("05_attk2_opt80_60_test")
FILES['80_opt'].append("05_attk2_opt80_80_test")


# AVG

nums = []
for id in ['20_avg', '40_avg', '50_avg', '60_avg', '80_avg']:
    for ff in FILES[id]:
        with open(DIR + ff, 'r') as f:
            print("Working on " + ff)
            lines = f.readlines()
            nums.append(lines[4].split()[2])
            f.close()

with open(DIR + "09-workers-data-avg.txt", 'w') as f:
    f.write('- - "20% Data Alteration" - "40% Data Alteration" - "50% Data Alteration" - "60% Data Alteration" - "80% Data Alteration "\n')
    f.write("20 20% " + nums[0] + " " + nums[1] + " " + nums[2] + " " + nums[3] + "\n")
    f.write("40 40% " + nums[4] + " " + nums[5] + " " + nums[6] + " " + nums[7] + "\n")
    f.write("50 50% " + nums[8] + " " + nums[9] + " " + nums[10] + " " + nums[11] + "\n")
    f.write("60 60% " + nums[12] + " " + nums[13] + " " + nums[14] + " " + nums[15] + "\n")
    f.write("80 80% " + nums[16] + " " + nums[17] + " " + nums[17] + " " + nums[19] + "\n")
    f.close()

# OPT

nums = []
for id in ['20_opt', '40_opt', '50_opt', '60_opt', '80_opt']:
    for ff in FILES[id]:
        with open(DIR + ff, 'r') as f:
            print("Working on " + ff)
            lines = f.readlines()
            nums.append(lines[4].split()[2])
            f.close()

with open(DIR + "09-workers-data-opt.txt", 'w') as f:
    f.write('- - "20% Data Alteration" - "40% Data Alteration" - "50% Data Alteration" - "60% Data Alteration" - "80% Data Alteration "\n')
    f.write("20 20% " + nums[0] + " " + nums[1] + " " + nums[2] + " " + nums[3] + "\n")
    f.write("40 40% " + nums[4] + " " + nums[5] + " " + nums[6] + " " + nums[7] + "\n")
    f.write("50 50% " + nums[8] + " " + nums[9] + " " + nums[10] + " " + nums[11] + "\n")
    f.write("60 60% " + nums[12] + " " + nums[13] + " " + nums[14] + " " + nums[15] + "\n")
    f.write("80 80% " + nums[16] + " " + nums[17] + " " + nums[17] + " " + nums[19] + "\n")
    f.close()