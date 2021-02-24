"""
Usage: 
    prepare.py --input=INPUT
"""
from docopt import docopt
import ast
import pickle
from tqdm import tqdm
arguments = docopt(__doc__)
# selected_ids = ["worker_{}".format(ii) for ii in range(0, 100)]

def load_object(file_path):
    with open(file_path, 'rb') as input:
        return pickle.load(input)
    
def main():
    file_name=arguments['--input']
    file_root="/Users/ehsan/data"
    file_root_output="{}/{}".format(file_root, "prepared_weights")
    file_path_data = "{}/{}/{}".format(file_root, "opt_weights", file_name)
    file_path_attacker = "{}/{}/{}".format(file_root, "attackers", file_name)
    
    file_path_output_a = "{}/{}_a.txt".format(file_root_output, file_name[:-4])
    file_path_output_n = "{}/{}_n.txt".format(file_root_output, file_name[:-4])

    attackers = load_object(file_path_attacker)
    aa = open(file_path_output_a, "w")
    nn = open(file_path_output_n, "w")
    count = 0
    all_weights = []
    with open(file_path_data, 'r') as f:
        for line in tqdm(f.readlines()):
            pos = line.find('{')
            line_dict = ast.literal_eval(line[-1*len(line)+pos:].strip())
            for ii, dd in line_dict.items():
                # if ii in selected_ids:
                # state = "A" if ii in attackers else "N"
                if ii in attackers:
                    state = "A"
                    aa.write("{} {} {}\n".format(ii[-1*len(ii)+7:], state, dd))
                else:
                    state = "N"
                    nn.write("{} {} {}\n".format(ii[-1*len(ii)+7:], state, dd))
        f.close()
    aa.close()
    nn.close()

    
if __name__ == '__main__':
    main()
