#!python3

import sys
from plom import parse_input, initialize, run

cli_args = sys.argv
if len(cli_args) > 1:
    input_path = cli_args[1]
else:
    input_path = 'input.txt'

args = parse_input(input_path)
solution_dict = initialize(**args)
run(solution_dict)