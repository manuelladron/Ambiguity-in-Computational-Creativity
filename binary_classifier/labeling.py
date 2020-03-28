import json
import pdb, sys, os, io

# events-example0.py
# Barebones timer, mouse, and keyboard events

from tkinter import *
from PIL import ImageTk, Image

####################################
# customize these functions
####################################
alphabet = 'abcdefghijklmnopqrstuvwxyz'
numbers = '0123456789'
def init(data):
    data.file_path = './data/design_dz-cleaned.json'
    data.titlesSeen = set()
    data.new_data = []
    data.example = None
    get_next_item(data)

def get_next_item(data):
    # Check if data copy exists, if it does, open it, if it doesn't, make one.

    with open(data.file_path) as f:
        output = json.loads(json.load(f))

    for example in output:
        # if there aren't any images, or any text, or the title has already been seen,
        # ignore it
        if (len(example['images']) > 0 and len(example['text']) > 0 and
           example['title'] not in data.titlesSeen and len(example['text']) < len(alphabet)
           and len(example['images']) < len(numbers)):
            # List the text by parts on one side, list the images on the other.
            data.example = example
            data.titlesSeen.add(example['title'])

            break

def mousePressed(event, data):
    # use event.x and event.y
    pass

def save_item(data):
    pass

def keyPressed(event, data):
    # Remove mispress
    if event.keysym in data.new_data:
        data.new_data.remove(event.keysym)
    # print(event.keysym)
    elif event.keysym in numbers:
        if int(event.keysym) < len(data.example['images']):
            data.new_data.append(event.keysym)
    elif event.keysym in alphabet:
        if alphabet.index(event.keysym) < len(data.example['text']):
            data.new_data.append(event.keysym)
    elif event.keysym == 'space':
        # TODO: Save Item
        data.new_data = []
        get_next_item(data)
    elif event.keysym == 'backspace':
        # TODO: Go back to previous item
        pass

def timerFired(data):
    pass


def drawImages(canvas, data):
    image_size = len(data.example['images'])
    base_height = int(data.height / image_size)

    data.imgs = []
    for index, image_name in enumerate(data.example['images']):
        name = './design/' + image_name
        image = Image.open(name)

        h_percent = (base_height/float(image.size[0]))
        wsize = int((float(image.size[0])*float(h_percent)))
        image = image.resize((base_height, wsize), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        data.imgs.append(image)
        canvas.create_image(720, index*base_height, anchor=NW, image=image)
        if str(index) in data.new_data:
            canvas.create_text(720 - 60, index*base_height, text='Active:'+str(index)+'.', font="Arial 14 bold", anchor=NW)
        else:
            canvas.create_text(720 - 20, index*base_height, text=str(index)+'.', font="Arial 14 bold", anchor=NW)

def drawSentences(canvas, data):
    t = ''
    for index, text in enumerate(data.example['text']):
        if alphabet[index] in data.new_data:
            t += 'Active: ' + alphabet[index] + '. ' + text + '\n\n'
        else:
            t += alphabet[index] + '. ' + text + '\n\n'
    text_size = '14' if  index <= 10 else '12'
    canvas.create_text(10, 10, text=t, font="Arial "+text_size+" bold", anchor=NW, width=data.width // 2)

def drawDirections(canvas, data):
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

