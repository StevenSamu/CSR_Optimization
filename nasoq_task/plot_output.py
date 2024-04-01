import matplotlib.pyplot as plt
import re
from subprocess import getoutput
import os 
import sys

bin_location = r"""build\Release\NASOQ-BIN.exe"""
test_file_dir = r"tests/Mpclib/"
epsilons = ["-3","-6","-9"] 
test_list = os.listdir(test_file_dir)
modes = ["NASOQ-tuned","NASOQ-fixed","NASOQ-auto"] #preset to add?
total_problems = len(test_list)

for epsilon in epsilons:
    for mode in modes:
        performance_ratio = []
        ratio_probs_solved = []
        probs_solved = 0
        min_solve_time = sys.maxsize
        for test in test_list: 
            test_file = fr"""tests\Mpclib\{test}"""
            test_name = test.split(".")[0]
            ret = getoutput([bin_location, '-i', test_file, '-e', epsilon, '-v', mode]).split('\n')
            for line in ret:
                if re.match(r'Status:', line):
                    data = int(line.split(": ")[-1]) # Extract the value
                    if (data == 1):
                        probs_solved += 1
                    ratio_probs_solved.append(probs_solved/total_problems) # Saving only values
                if re.match(r'Time', line):
                    data = float(line.split(": ")[-1]) # Extract the value
                    if (data < min_solve_time):
                        min_solve_time = data
                    performance_ratio.append(data/min_solve_time) 
        plt.plot(ratio_probs_solved, performance_ratio, label= mode)
    plt.ylabel('Performance Ratio rp,s') # ratio of the current solve time to the minimum solve time
    plt.xlabel('Ratio of problems solved')
    plt.title(f"Epsilon: 10e{epsilon}")
    plt.legend()
    plt.savefig(fr'test_results\{epsilon}.png')
    plt.clf()
