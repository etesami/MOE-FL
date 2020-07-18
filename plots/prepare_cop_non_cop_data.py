#!/usr/bin/python

DIR = "data_tmp/"

FILES = {}
FILES['20'] = []
FILES['50'] = []
FILES['80'] = []

FILES['20'].append("04_attk2_avg20_40_test")
FILES['20'].append("05_attk2_opt20_40_test")
FILES['20'].append("04_attk2_avg20_80_test")
FILES['20'].append("05_attk2_opt20_80_test")

FILES['50'].append("04_attk2_avg50_40_test")
FILES['50'].append("05_attk2_opt50_40_test")
FILES['50'].append("04_attk2_avg50_80_test")
FILES['50'].append("05_attk2_opt50_80_test")

FILES['80'].append("04_attk2_avg80_40_test")
FILES['80'].append("05_attk2_opt80_40_test")
FILES['80'].append("04_attk2_avg80_80_test")
FILES['80'].append("05_attk2_opt80_80_test")

nums = []
for id in ['20', '50', '80']:
    for ff in FILES[id]:
        with open(DIR + ff, 'r') as f:
            print("Working on " + ff)
            lines = f.readlines()
            nums.append(lines[4].split()[2])
            f.close()

with open(DIR + "09-cooperative-non-cop-test.txt", 'w') as f:
    f.write('- - "40% Data Alteration (AVG)" - "40% Data Alteration (Weighted AVG)" - "80% Data Alteration (AVG)" - "80% Data Alteration (Weighted AVG)"\n')
    f.write("20% " + nums[0] + " - " + nums[1] + " - " + nums[2] + " - " + nums[3] + " -\n")
    f.write("50% " + nums[4] + " - " + nums[5] + " - " + nums[6] + " - " + nums[7] + " -\n")
    f.write("80% " + nums[8] + " - " + nums[9] + " - " + nums[10] + " - " + nums[11] + " -\n")
    f.close()


'''
- - "40% Data Alteration (AVG)" - "40% Data Alteration (Weighted AVG)" - "80% Data Alteration (AVG)" - "80% Data Alteration (Weighted AVG)"
20 20 30 40 50
40 20 30 40 50
60 20 30 40 50
80 20 30 40 50
'''