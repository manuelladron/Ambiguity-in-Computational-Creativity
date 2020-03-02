import sys, os
from dz import dz

def main(file_path, website):
    if website == 'dz':
        runner = dz(file_path)
        runner.run()
    # Add more websites here and make their corresponding classes


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('*'*20, '\nIncorrect number of arguemenets...')
        print('Example: python3 main.py filePath.json dz\n'+'*'*20)
    elif not os.path.exists(sys.argv[-2]):
        print('File %s does not exist.' % sys.argv[-2])
    else:
        main(sys.argv[-2], sys.argv[-1])

