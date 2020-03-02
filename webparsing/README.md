### dz
Data
```
{
    title: [String],
    text: [List of Strings] Quotes from article,
    images: [List of Strings] Files paths to images
}
```
Loading the output to use
```
with open(file_path) as f:
    output = json.loads(json.load(f))
```
