## Running the data labeling

```
python3 labeling.py ./data/file_name-cleaned.json ./image_folder/
```

### Usage
* [q] selects/deselects all text
* [w] selects/deselects all images
* [space] continues to the next example with prompts if you didn't input data correctly. When [space] is pressed the current example is saved.
* [BackSpace] goes to the previous example, this is persistent across runs, so you could go back and see all of the examples you've labeled before. When BackSpace is pressed the current example is saved.
* All of the new data is saved into './data/file_name-cleaned-tagged.json' The data is the same, other than two new keys specifying which images/texts we have "tagged." So it looks like:

```
{
    title: [String],
    text: [List of Strings] Quotes from article,
    images: [List of Strings] Files paths to images,
    image-tags: [List of Strings] (0-9 depending on what you tagged),
    text-tags: [List of Strings] (a-p depending on what you tagged, can be turned into indexes)
}
```
