Organizes the dataset into the folowing fields per article:

                cleaned_example['text'] = One cleaned string 
                cleaned_example['images'] = path to images
                cleaned_example['title'] = title
                cleaned_example['date'] = date
                cleaned_example['quotes'] = Text in quotes
                cleaned_example['no_quotes'] = Text not in quotes
                cleaned_example['adj_quotes'] = Adjectives found in text in quotes
                cleaned_example['adj_no_quotes'] = Adjectives found in text not in quotes

Returns a json file containing a list of articles. Each article is a dictionary with the above fields. 
The last element of this json file is a dictionary containing the counts of adjectives used by reporters and authors 
