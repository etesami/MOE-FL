#!/usr/bin/python

DIR = "data_tmp/"

FILES = []
FILES.append("02_attk1_avg20_test")
FILES.append("02_attk1_avg40_test")
FILES.append("02_attk1_avg50_test")
FILES.append("02_attk1_avg60_test")
FILES.append("02_attk1_avg80_test")

FILES.append("03_attk1_opt20_test")
FILES.append("03_attk1_opt40_test")
FILES.append("03_attk1_opt50_test")
FILES.append("03_attk1_opt60_test")
FILES.append("03_attk1_opt80_test")

nums = []
for ff in FILES:
    with open(DIR + ff, 'r') as f:
        print("Working on " + ff)
        lines = f.readlines()
        nums.append(lines[4].split('"')[2].strip())
        f.close()

with open(DIR + "09-non-cooperative-test.txt", 'w') as f:
    f.write('- - AVG - "Proposed Approach"\n')
    f.write("20% " + nums[0] + " AVG " + nums[5] + " Proposed Approach\n")
    f.write("40% " + nums[1] + " AVG " + nums[6] + " Proposed Approach\n")
    f.write("50% " + nums[2] + " AVG " + nums[7] + " Proposed Approach\n")
    f.write("60% " + nums[3] + " AVG " + nums[8] + " Proposed Approach\n")
    f.write("80% " + nums[4] + " AVG " + nums[9] + " Proposed Approach\n")