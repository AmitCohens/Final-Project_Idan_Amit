import argparse as arg
import os
import subprocess


parser = arg.ArgumentParser()

parser.add_argument('-r')

args = parser.parse_args()

if __name__ == '__main__':
    threads = []
    number = args.r
    number = int(number)
    if not os.path.exists(f"{number}-{number+999999}"):
        os.mkdir(f"{number}-{number+999999}")
    for i in range(number, number + 999999, 333333):
        subprocess.call(f'start cmd /c python C:\\Users\\Amit_Idan\\Final-Project\\tester.py -run {i} -nums {number}', shell=True)
