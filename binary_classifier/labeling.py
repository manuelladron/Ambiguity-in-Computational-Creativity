import json
import pdb, sys, os, io

# events-example0.py
# Barebones timer, mouse, and keyboard events

from tkinter import *
from PIL import ImageTk, Image

####################################
# customize these functions
####################################

def init(data):
    data.file_path = './data/design_dz-cleaned.json'
    data.titlesSeen = set()
    data.example = None
    get_next_item(data)

def get_next_item(data):
    with open(data.file_path) as f:
        output = json.loads(json.load(f))

    for example in output:
        # if there aren't any images, or any text, or the title has already been seen,
        # ignore it
        if (len(example['images']) > 0 and len(example['text']) > 0 and
           example['title'] not in data.titlesSeen):
            print(example)
            # List the text by parts on one side, list the images on the other.
            data.example = example
            # for index, image_name in enumerate(data.example['images']):
            #     name = 'design/' + image_name
            #     print(name)
                # data.example[index] = PhotoImage(file=name)

            break
            # 1) Write to new seen list
            # 2) Save sentences and images to new json, with label
            # u.save_json(new_file_name, data)
    # load data.xyz as appropriate

def mousePressed(event, data):
    # use event.x and event.y
    pass

def keyPressed(event, data):
    # use event.char and event.keysym
    pass

def timerFired(data):
    pass

def redrawAll(canvas, data):
    image_size = len(data.example['images'])
    base_height = int(data.height / image_size)

    data.imgs = []
    for index, image_name in enumerate(data.example['images']):
        name = './design/' + image_name
        image = Image.open(name)

        h_percent = (base_height/float(image.size[1]))
        wsize = int((float(image.size[0])*float(h_percent)))
        image = image.resize((base_height, wsize), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        data.imgs.append(image)
        canvas.create_image(540, index*base_height, anchor=NW, image=image)

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

