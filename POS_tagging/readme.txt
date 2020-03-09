The structure of the code is based on Webparsing. 

To load json, you can use this function:

 def load_json(file_path):
     with open(file_path) as f:
         data = json.load(f)
     f.close()
     return data
