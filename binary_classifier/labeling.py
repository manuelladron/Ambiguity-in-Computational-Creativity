import json
import pdb, sys, os, io

from tkinter import *
from PIL import ImageTk, Image


ALPHABET = 'abcdefghijklmnop'
NUMBERS = '0123456789'

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

def init(data):
    data.file_path = './data/design_dz-cleaned.json'
    data.titlesSeen = set()


    data.new_letters = []
    data.new_images = []

    data.example = None
    data.output = None
    get_next_item(data)

def get_next_item(data):
    # Check if data copy exists, if it does, open it, if it doesn't, make one.
    if data.output is None:
        with open(data.file_path) as f:
            data.output = json.loads(json.load(f))

    # Iterate through current data.output
    i = 0
    for example in data.output:
        # Ignore examples without images/text
        if (len(example['images']) > 0 and len(example['text']) > 0):
            # Ignore examples already viewed
            if example['title'] not in data.titlesSeen:
                # Ignore examples that have too many images/texts
                if (len(example['text']) <= 16 and len(example['images']) <= len(NUMBERS)):
                    data.example = example
                    data.titlesSeen.add(example['title'])
                    break
        i += 1

    # TODO: Remove the next i elements from list that we've already viewed.

def mousePressed(event, data):
    # use event.x and event.y
    pass

def save_item(data):
    pass

def keyPressed(event, data):
    # If an element is already active, remove it by pressing the key again
    if event.keysym in data.new_letters or event.keysym in data.new_images:
        data.new_data.remove(event.keysym)

    # Add an element to the list of saved items
    elif event.keysym in NUMBERS:
        if int(event.keysym) < len(data.example['images']):
            data.new_images.append(event.keysym)
    elif event.keysym in ALPHABET:
        if ALPHABET.index(event.keysym) < len(data.example['text']):
            data.new_letters.append(event.keysym)

    # TODO: Save an item
    elif event.keysym == 'space':
        data.new_images = []
        data.new_letters = []
        get_next_item(data)

    # Select/ Deselect all images
    elif event.keysym == 'q':
        if isTextSelected(data):
            data.new_letters = []
        else:
            data.new_letters = [ALPHABET[i] for i in range(len(data.example['text']))]

    # Select/ Deselect all images
    elif event.keysym == 'w':
        if isImageSelected(data):
            data.new_images = []
        else:
            data.new_images = [NUMBERS[i] for i in range(len(data.example['images']))]

    # TODO: Revert to a previous item
    elif event.keysym == 'backspace':
        pass

def timerFired(data):
    pass


def drawImages(canvas, data):
    image_size = len(data.example['images'])
    base_height = int(data.height / image_size)
    image_left = 3 * data.width // 4 + 20

    data.imgs = []
    for index, image_name in enumerate(data.example['images']):
        name = './design/' + image_name
        image = Image.open(name)

        h_percent = (base_height/float(image.size[0]))
        wsize = int((float(image.size[0])*float(h_percent)))
        image = image.resize((base_height, wsize), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        data.imgs.append(image)
        canvas.create_image(image_left, index*base_height, anchor=NW, image=image)
        if str(index) in data.new_images:
            canvas.create_text(image_left - 60, index*base_height, text='Active:'+str(index)+'.', font="Arial 14 bold", anchor=NW)
        else:
            canvas.create_text(image_left - 20, index*base_height, text=str(index)+'.', font="Arial 14 bold", anchor=NW)

def drawSentences(canvas, data):
    t = ''
    for index, text in enumerate(data.example['text']):
        if ALPHABET[index] in data.new_letters:
            t += 'Active: ' + ALPHABET[index] + '. ' + text + '\n\n'
        else:
            t += ALPHABET[index] + '. ' + text + '\n\n'
    text_size = '14' if  index <= 10 else '12'
    canvas.create_text(10, 10, text=t, font="Arial "+text_size, anchor=NW, width= 2 * data.width // 3)

def drawDirections(canvas, data):
    # Title
    title_text = 'Title: ' + data.example['title']
    canvas.create_text(10, data.height - 100, text=title_text, font="Arial 14 italic", anchor=NW)

    # Select all text/ deselect all text
    select_text = '[q] Select' if not isTextSelected(data) else '[q] Deselect'
    select_text += ' all text'
    canvas.create_text(10, data.height - 80, text=select_text, font="Arial 14", anchor=NW)

    # Select all images/ deselect all images
    select_images = '[w] Select' if not isImageSelected(data) else '[q] Deselect'
    select_images += ' all images'
    canvas.create_text(10, data.height - 60, text=select_images, font="Arial 14", anchor=NW)

    # Save

    # Back


    pass

def redrawAll(canvas, data):
    drawImages(canvas, data)
    drawSentences(canvas, data)
    drawDirections(canvas, data)

####################################
# use the run function as-is
####################################

def run(width=300, height=300):
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


run(1080, 720)

