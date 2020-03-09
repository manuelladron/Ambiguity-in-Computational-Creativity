Samples of json files have another key: "adjectives". 
Each json file has a new element in the list at idx -1, which is the adjectives BOW (unique elements).

The structure of the code is based on Webparsing. 

To load json, you can use this function:

 def load_json(file_path):
     with open(file_path) as f:
         data = json.load(f)
     f.close()
     return data
