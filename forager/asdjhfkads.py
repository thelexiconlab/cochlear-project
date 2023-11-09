import os 

path = '/Users/mkang2/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/cochlear-project/forager/new_data.txt'

if os.path.exists(path) == False:
    ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path