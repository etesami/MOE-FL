#!/usr/bin/python

DIR = "data_tmp/"

FILES = {}
FILES['20'] = []
FILES['40'] = []
FILES['50'] = []
FILES['60'] = []
FILES['80'] = []

FILES['20'].append("04_attk2_avg20_20_test")
FILES['20'].append("05_attk2_opt20_20_test")
FILES['20'].append("04_attk2_avg20_40_test")
FILES['20'].append("05_attk2_opt20_40_test")
FILES['20'].append("04_attk2_avg20_60_test")
FILES['20'].append("05_attk2_opt20_60_test")
FILES['20'].append("04_attk2_avg20_80_test")
FILES['20'].append("05_attk2_opt20_80_test")

FILES['40'].append("04_attk2_avg40_20_test")
FILES['40'].append("05_attk2_opt40_20_test")
FILES['40'].append("04_attk2_avg40_40_test")
FILES['40'].append("05_attk2_opt40_40_test")
FILES['40'].append("04_attk2_avg40_60_test")
FILES['40'].append("05_attk2_opt40_60_test")
FILES['40'].append("04_attk2_avg40_80_test")
FILES['40'].append("05_attk2_opt40_80_test")

FILES['50'].append("04_attk2_avg50_20_test")
FILES['50'].append("05_attk2_opt50_20_test")
FILES['50'].append("04_attk2_avg50_40_test")
FILES['50'].append("05_attk2_opt50_40_test")
FILES['50'].append("04_attk2_avg50_60_test")
FILES['50'].append("05_attk2_opt50_60_test")
FILES['50'].append("04_attk2_avg50_80_test")
FILES['50'].append("05_attk2_opt50_80_test")

FILES['60'].append("04_attk2_avg60_20_test")
FILES['60'].append("05_attk2_opt60_20_test")
FILES['60'].append("04_attk2_avg60_40_test")
FILES['60'].append("05_attk2_opt60_40_test")
FILES['60'].append("04_attk2_avg60_60_test")
FILES['60'].append("05_attk2_opt60_60_test")
FILES['60'].append("04_attk2_avg60_80_test")
FILES['60'].append("05_attk2_opt60_80_test")

FILES['80'].append("04_attk2_avg80_20_test")
FILES['80'].append("05_attk2_opt80_20_test")
FILES['80'].append("04_attk2_avg80_40_test")
FILES['80'].append("05_attk2_opt80_40_test")
FILES['80'].append("04_attk2_avg80_60_test")
FILES['80'].append("05_attk2_opt80_60_test")
FILES['80'].append("04_attk2_avg80_80_test")
FILES['80'].append("05_attk2_opt80_80_test")

nums = []
for id in ['20', '40', '50', '60', '80']:
    for ff in FILES[id]:
        with open(DIR + ff, 'r') as f:
            print("Working on " + ff)
            lines = f.readlines()
            nums.append(lines[4].split('"')[2].strip())
            f.close()

with open(DIR + "09-cooperative-test.txt", 'w') as f:
    f.write('- - "20% Data Alteration (AVG)" - "20% Data Alteration (Proposed Approach)" - "40% Data Alteration (AVG)" - "40% Data Alteration (Proposed Approach)" - "60% Data Alteration (AVG)" - "60% Data Alteration (Proposed Approach)" - "80% Data Alteration (AVG)" - "80% Data Alteration (Proposed Approach)"\n')
    f.write("20% " + nums[0] + " - " + nums[1] + " - " + nums[2] + " - " + nums[3] + " - " + nums[4] + " - " + nums[5] + " - " + nums[6] + " - " + nums[7] + " \n")

    f.write("40% " + nums[8] + " - " + nums[9] + " - " + nums[10] + " - " + nums[11] + " - " + nums[12] + " - " + nums[13] + " - " + nums[14] + " - " + nums[15] + " \n")

    f.write("50% " + nums[16] + " - " + nums[17] + " - " + nums[18] + " - " + nums[19] + " - " + nums[20] + " - " + nums[21] + " - " + nums[22] + " - " + nums[23] + " \n")

    f.write("60% " + nums[24] + " - " + nums[25] + " - " + nums[26] + " - " + nums[27] + " - " + nums[28] + " - " + nums[29] + " - " + nums[30] + " - " + nums[31] + " \n")

    f.write("80% " + nums[32] + " - " + nums[33] + " - " + nums[34] + " - " + nums[35] + " - " + nums[36] + " - " + nums[37] + " - " + nums[38] + " - " + nums[39] + " \n")

    f.close()
