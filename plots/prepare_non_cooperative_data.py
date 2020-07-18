#!/usr/bin/python

DIR = "data_tmp/"

FILES = []
FILES.append("02_attk1_avg20_test")
FILES.append("02_attk1_avg40_test")
FILES.append("02_attk1_avg60_test")
FILES.append("02_attk1_avg80_test")

FILES.append("03_attk1_opt20_test")
FILES.append("03_attk1_opt40_test")
FILES.append("03_attk1_opt60_test")
FILES.append("03_attk1_opt80_test")

nums = []
for ff in FILES:
    with open(DIR + ff, 'r') as f:
        print("Working on " + ff)
        lines = f.readlines()
        nums.append(lines[4].split()[2])
        f.close()

with open(DIR + "09-non-cooperative-test.txt", 'w') as f:
    f.write("- - AVG - Weighted AVG\n")
    f.write("20% " + nums[0] + " AVG " + nums[4] + " Weighted AVG\n")
    f.write("40% " + nums[1] + " AVG " + nums[5] + " Weighted AVG\n")
    f.write("60% " + nums[2] + " AVG " + nums[6] + " Weighted AVG\n")
    f.write("80% " + nums[3] + " AVG " + nums[7] + " Weighted AVG\n")