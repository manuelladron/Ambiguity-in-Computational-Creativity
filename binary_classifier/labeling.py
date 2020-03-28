import json
import pdb, sys, os, io
import copy

from tkinter import *
from PIL import ImageTk, Image
from utils import Utils


ALPHABET = 'abcdefghijklmnop'
NUMBERS = '0123456789'
isTesting = True

#########################################################
#########################################################

'''
Returns true if any text is selected
'''
def isTextSelected(data):
    return len(data.new_letters) != 0

'''
Returns true is anyimage is selected
'''
def isImageSelected(data):
    return len(data.new_images) != 0

'''
Sets the index for the next item in the valid examples
'''
def get_next_item(data):
    data.i += 1
    if data.i == len(data.examples):
        print('You have seen all of the examples!')
        exit(0)

'''
Sets the index to be looking at the previous item
'''
def get_prev_item(data):
    if (data.i == 0):
        print('You cannot see any previous examples, you are at example 0.')
    else:
        data.i -= 1

'''
Sets data.isPromptOpen and data.prompt for different cases of [space] press
'''
def isExampleCompleteChecks(data):
    nothingSelected = 'Selecting nothing will remove this example from the data.\n[space] to continue.\n[backspace] go back and fix.'
    if data.isPromptOpen == True and data.prompt == nothingSelected:
        data.isPromptOpen = False
        return

    elif len(data.new_images) == 0 and len(data.new_letters) != 0:
        data.prompt = 'You have text selected but no images.\n[backspace] go back and fix.'
        data.isPromptOpen = True
        return

    elif len(data.new_letters) == 0 and len(data.new_images) != 0:
        data.prompt = 'You have images selected but not text.\n[backspace] go back and fix.'
        data.isPromptOpen = True
        return

    elif len(data.new_images) == 0 and len(data.new_letters) == 0:
        data.prompt = nothingSelected
        data.isPromptOpen = True
        return

    data.isPromptOpen = False

#########################################################
#########################################################

def init(data):
    data.file_path_tagged = '%s-tagged' % data.file_path.split('.json')[0] + '.json'
    data.u = Utils()

    data.new_letters = []
    data.new_images = []

    data.isPromptOpen = False
    data.prompt = None

    data.output = None # json data
    data.i = 0         # index of current example
    data.examples = [] # all valid viewable examples from json data
    data.examples_tagged = []
    get_examples(data)

'''
Iterates through all of the data and makes a list data.examples of
all of the valid examples
'''
def get_examples(data):
    with open(data.file_path) as f:
        data.output = json.loads(json.load(f))

    if os.path.exists(data.file_path_tagged):
        with open(data.file_path_tagged) as f:
            data.examples_tagged = json.loads(json.load(f))

    # Iterate through current data.output and make list of new examples
    for example in data.output:
        # Ignore examples without images/text
        if (len(example['images']) > 0 and len(example['text']) > 0):
            # Ignore examples that have too many images/texts
            if (len(example['text']) <= 16 and len(example['images']) <= len(NUMBERS)):
                example['image-tags'] = []
                example['text-tags'] = []
                data.examples.append(example)

    # Parse out any examples already tagged
    tagged_examples_titles = set([example['title'] for example in data.examples_tagged])
    tagged_examples = [example['title'] for example in data.examples_tagged]
    # indexs_to_remove = []
    for index, example in enumerate(data.examples):
        if example['title'] in tagged_examples_titles:
            tagged_i = tagged_examples.index(example['title'])
            data.examples[index]['image-tags'] = data.examples_tagged[tagged_i]['image-tags']
            data.examples[index]['text-tags'] = data.examples_tagged[tagged_i]['text-tags']
            data.i = index + 1
            # indexs_to_remove.append(index)
    # for index in sorted(indexs_to_remove, reverse=True):
        # del data.output[index]

def save_item(data):
    data.examples[data.i]['image-tags'] = copy.copy(data.new_images)
    data.examples[data.i]['text-tags'] = copy.copy(data.new_letters)

    # if the example is already in data.examples_tagged, then update it
    # otherwise, append it
    example_titles = [example['title'] for example in data.examples_tagged]
    current_example_title = data.examples[data.i]['title']
    if current_example_title in example_titles:
        i = example_titles.index(current_example_title)
        data.examples_tagged[i] = data.examples[data.i]
    else:
        data.examples_tagged.append(data.examples[data.i])

    data.u.save_json(data.file_path_tagged, data.examples_tagged)

#########################################################
#########################################################

def keyPressed(event, data):
    # Revert to a previous item
    if event.keysym == 'BackSpace':
        if data.isPromptOpen:
            data.isPromptOpen = False
        else:
            save_item(data)
            get_prev_item(data)
            data.new_images = copy.copy(data.examples[data.i]['image-tags'])
            data.new_letters = copy.copy(data.examples[data.i]['text-tags'])

    # Continue onto a new item
    elif event.keysym == 'space':
        isExampleCompleteChecks(data)
        if data.isPromptOpen == False:
            save_item(data)
            get_next_item(data)
            data.new_images = copy.copy(data.examples[data.i]['image-tags'])
            data.new_letters = copy.copy(data.examples[data.i]['text-tags'])
        print("Tagged Examples: ", len(data.examples_tagged))

    if data.isPromptOpen:
        return

    # If an element is already active, remove it by pressing the key again
    elif event.keysym in data.new_letters:
        data.new_letters.remove(event.keysym)
    elif event.keysym in data.new_images:
        data.new_images.remove(event.keysym)

    # Add an element to the list of saved items
    elif event.keysym in NUMBERS:
        if int(event.keysym) < len(data.examples[data.i]['images']):
            data.new_images.append(event.keysym)
    elif event.keysym in ALPHABET:
        if ALPHABET.index(event.keysym) < len(data.examples[data.i]['text']):
            data.new_letters.append(event.keysym)

    # Select/ Deselect all images
    elif event.keysym == 'q':
        if isTextSelected(data):
            data.new_letters = []
        else:
            data.new_letters = [ALPHABET[i] for i in range(len(data.examples[data.i]['text']))]

    # Select/ Deselect all images
    elif event.keysym == 'w':
        if isImageSelected(data):
            data.new_images = []
        else:
            data.new_images = [NUMBERS[i] for i in range(len(data.examples[data.i]['images']))]



#########################################################
#########################################################

def drawImages(canvas, data):
    image_size = len(data.examples[data.i]['images'])
    base_height = int(data.height / image_size)
    image_left = 3 * data.width // 4 + 20

    data.imgs = []
    for index, image_name in enumerate(data.examples[data.i]['images']):
        # Get and draw image
        name = data.image_folder_path + image_name
        image = Image.open(name)
        h_percent = (base_height/float(image.size[0]))
        wsize = int((float(image.size[0])*float(h_percent)))
        image = image.resize((base_height, wsize), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        data.imgs.append(image)
        canvas.create_image(image_left, index*base_height, anchor=NW, image=image)

        # Get and draw text
        text = 'ACTIVE: '+str(index) if str(index) in data.new_images else str(index)
        canvas.create_text(image_left - 5, index*base_height + 10, text=text, font="Arial 14", anchor=NE)

def drawSentences(canvas, data):
    t = ''
    for index, text in enumerate(data.examples[data.i]['text']):
        if ALPHABET[index] in data.new_letters:
            t += 'ACTIVE: ' + ALPHABET[index] + '. ' + text + '\n\n'
        else:
            t += ALPHABET[index] + '. ' + text + '\n\n'
    text_size = '14' if  index <= 10 else '12'
    canvas.create_text(10, 10, text=t, font="Arial "+text_size, anchor=NW, width= 2 * data.width // 3)

def drawDirections(canvas, data):
    # Title
    title_text = 'Title: ' + data.examples[data.i]['title']
    canvas.create_text(10, data.height - 100, text=title_text, font="Arial 14 italic", anchor=NW)

    # Select all text/ deselect all text
    select_text = '[q] Select' if not isTextSelected(data) else '[q] Deselect'
    select_text += ' all text'
    canvas.create_text(10, data.height - 80, text=select_text, font="Arial 14", anchor=NW)

    # Select all images/ deselect all images
    select_images = '[w] Select' if not isImageSelected(data) else '[w] Deselect'
    select_images += ' all images'
    canvas.create_text(10, data.height - 60, text=select_images, font="Arial 14", anchor=NW)

    # Save
    save_text = '[space] Save an example and continue'
    canvas.create_text(10, data.height - 40, text=save_text, font="Arial 14", anchor=NW)

    # Back
    back_text = '[backspace] Go back one example'
    canvas.create_text(10, data.height - 20, text=back_text, font="Arial 14", anchor=NW)

def drawPrompt(canvas, data):
    if data.isPromptOpen:
        left, top = data.width // 3, data.height // 3
        color = "#%02x%02x%02x" % (210, 245, 255)
        canvas.create_rectangle(left - 10, top, 2 * data.width // 3 + 20, 2 * data.height // 3, fill=color)
        canvas.create_text(data.width//2, data.height //2, text=data.prompt, font="Arial 14",)

def redrawAll(canvas, data):
    drawImages(canvas, data)
    drawSentences(canvas, data)
    drawDirections(canvas, data)
    drawPrompt(canvas, data)

####################################
# Unused event functions
####################################

def mousePressed(event, data):
    # use event.x and event.y
    pass

def timerFired(data):
    pass

####################################
# use the run function as-is
####################################

def run(width, height, file_path, folder_path):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()

    data.width = width
    data.height = height
    data.file_path = file_path
    data.image_folder_path = folder_path

    data.timerDelay = 100 # milliseconds
    init(data)

    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('*'*20)
        print('Incorrect number of arguemenets...')
        print('Example: python3 labeling.py ./data/design_dz-cleaned.json ./design/')
        print('*'*20)
    elif not os.path.exists(sys.argv[-2]):
        print('File %s does not exist.' % sys.argv[-2])
    elif not os.path.exists(sys.argv[-1]):
        print('Folder %s does not exist.' % sys.argv[-1])
    else:
        run(1080, 720, sys.argv[-2], sys.argv[-1])

