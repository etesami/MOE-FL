import os

DIR = "./data_tmp/"
FILES = ['01_avg_test', '01_opt_test', ]
for i in [20, 40, 50, 60, 80]:
    FILES.append("02_attk1_avg" + str(i) + "_test")
    FILES.append("03_attk1_opt" + str(i) + "_test")
    for j in [20, 40, 60, 80, 100]:
        FILES.append("04_attk2_avg" + str(i) + "_" + str(j) +"_test")
        FILES.append("05_attk2_opt" + str(i) + "_" + str(j) +"_test")

# for ff in FILES:
#     print(ff)
# FILES.remove("05_attk2_opt40_80_test")
# FILES.remove("04_attk2_avg40_80_test")
for ff in FILES:
    ff_tmp = ff + "_tmp"
    with open(DIR + ff_tmp, 'w') as fw:
        with open(DIR + ff, 'r') as f:
            print(DIR + ff)
            lines = f.readlines()
            for ll in lines:
                acc = ll.strip().split('"')[0] + '"' + ll.strip().split('"')[1].split("%")[0] + '%" ' + ll.strip().split('"')[1].split("%")[1].strip()
                fw.write(acc + "\n")
            f.close()
        fw.close()
    os.system("mv " + DIR + ff_tmp + " " + DIR + ff)
           