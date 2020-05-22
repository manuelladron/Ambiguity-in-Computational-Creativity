run: python3 labeling_adj.py cleaned_json_file_path

Opens up a simple GUI with instructions for using it 
You can close it and come back without losing track of the labeling

Stores a new json file in the same directory as the input json file. It is a copy of the original json file with an 
element in the list at position idx = -1. 
This contains a list with all adjectives tagged in the form [adjective name, source, relevant, ambiguity]

