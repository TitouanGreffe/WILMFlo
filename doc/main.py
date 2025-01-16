import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#import sys
#sys.path.append('/home/gtitouan/scratch/RESEDA/modulefiles/')

#from Reseda_sim import reseda_world
from WILMflo import WILMFlo_world
import pandas as pd
import time

## Change in_folder_path accordingly to your path on your PC/Mac
in_folder_path = "/home/gtitouan/scratch/RESEDA/Input_data/"
in_file = "SI_1_Inputs_WILMFlo.xlsx"

## Create a folder named "Output_data" on your PC and change output_folder accordingly to your path on your PC/Mac
output_folder = "/home/gtitouan/scratch/RESEDA/Output_data/"
output_file = "output.xlsx"

demand = ["DLS","STEPS"]
prod = ["IEA_NZ","STEPS"]

'''
To simulate the scenario Decent Living Standards in a Net Zero world, use:
obj = WILMFlo_world(in_folder_path, in_file, demand[0], prod[0],output_folder, output_file)

To simulate the scenario STEPS, use:
obj = WILMFlo_world(in_folder_path, in_file, demand[1], prod[1],output_folder, output_file)

To simulate the scenario Net Zero, use:
obj = WILMFlo_world(in_folder_path, in_file, demand[1], prod[0],output_folder, output_file)

'''

obj = WILMFlo_world(in_folder_path, in_file, demand[1], prod[1],output_folder, output_file)

obj.solve_scenario()



