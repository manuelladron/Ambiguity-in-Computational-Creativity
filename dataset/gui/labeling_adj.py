import json
import pdb, sys, os, io
import copy
import random

from tkinter import *

ALPHABET = 'abc'
NUMBERS = '012'
isTesting = True

##############################

'''
Returns true if any text is selected
'''
def isRelevantSelected(data):
    return data.relevant != None

'''
Returns true is anyimage is selected
'''
def isAmbiguitySelected(data):
    return data.ambiguity != None

'''
Sets the index for the next item in the valid examples
'''
def get_next_item(data):
    data.i += 1
    if data.i == len(data.adjectives):
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

    elif data.ambiguity == None and data.relevant != None:
        data.prompt = 'You have entered relevant but no ambiguity.\n[backspace] go back and fix.'
        data.isPromptOpen = True
        return

    elif data.relevant == None and data.ambiguity != None:
        data.prompt = 'You have entered ambiguity but not relevant.\n[backspace] go back and fix.'
        data.isPromptOpen = True
        return

    elif data.ambiguity == None and data.relevant == None:
        data.prompt = nothingSelected
        data.isPromptOpen = True
        return

    data.isPromptOpen = False

#########################################################
#########################################################

def init(data):
    data.file_path_tagged = '%s-tagged' % data.file_path.split('.json')[0] + '.json'

    # context
    name = data.file_path.split("/")[-1]
    id = name.find("_")
    data.context = name[:id]
    if data.context == 'wearable':
        data.context = 'wearable technology'

    data.relevant  = None
    data.ambiguity = None

    data.isPromptOpen = False
    data.prompt = None

    data.output = None        # json data
    data.json_tagged = None   # json tagged

    data.i = 0         # index of current example
    data.adjectives = [] # all valid viewable examples from json data
    data.adjectives_tagged = []
    get_examples(data)


'''
Iterates through all of the data and makes a list data.examples of
all of the valid examples
'''

def get_examples(data):
    # open json file
    f = open(data.file_path)
    data.output = json.load(f)
    f.close()

    # check if the new tagged file is opened
    if os.path.exists(data.file_path_tagged):
        with open(data.file_path_tagged) as f:
            data.json_tagged = json.load(f)

    # Get dictionary of the file
    adj_dictionary = data.output[-1]
    reporters = adj_dictionary['reporters']
    authors = adj_dictionary['authors']

    # Sort both dictionaries
    reporters = sorted(reporters.items(), key=lambda reporters: reporters[1], reverse=True)
    authors = sorted(authors.items(), key=lambda authors: authors[1], reverse=True)

    all_adjectives = []
    for i in range(len(reporters)):
        adj = reporters[i][0]
        all_adjectives.append((adj, 'r'))

    # if adjective already in list, update source 'r' becomes 'b' (reporter becomes both), else append
    for j in range(len(authors)):
        adj = authors[j][0]

        if (adj, 'r') in all_adjectives:            # update
            id = all_adjectives.index((adj, 'r'))
            all_adjectives[id] = ((adj, 'b'))
        else:                                       # append
            all_adjectives.append((adj, 'a'))

    # Shuffle list
    random.shuffle(all_adjectives)

    # Get tagged adjectives. important: tagged adjectives are in the form (adj, source, relevant, ambiguity), while
    # important:                        adjectives in data.adjectives are in the form (adj, source)

    if data.json_tagged != None:
        data.adjectives_tagged = data.json_tagged[-1]

    already_tagged = [adj[0] for adj in data.adjectives_tagged]
    print("already tagged: ")
    print(already_tagged)

    # Iterate over the list and store adjectives in data class if not already tagged.
    for adjective in all_adjectives:
        if adjective[0] not in already_tagged:
            data.adjectives.append(adjective)

    print("all_adjectives: ", len(all_adjectives))
    print("data.adjectives: ", len(data.adjectives))

def save_item(data):
    # information to store in the new dictionary
    adj_text = data.adjectives[data.i][0]
    source =   data.adjectives[data.i][1]
    relevant = data.relevant
    ambiguity = data.ambiguity

    # if the example is already in data.examples_tagged, then update it
    # otherwise, append it

    # create a list with the names only to use it for checking whether it's already there
    only_adj = [adj[0] for adj in data.adjectives_tagged]

    if adj_text in only_adj:
        idx = only_adj.index(adj_text)
        data.adjectives_tagged[idx] = (adj_text, source, relevant, ambiguity)
    else:
        data.adjectives_tagged.append((adj_text, source, relevant, ambiguity))

    # if json_tagged equals none means that is the first example and we need to append a list to the json file,
    # otherwise, we just update the last element of the json file (the list with the new info)

    if data.json_tagged == None:
        data.json_tagged = data.output
        data.json_tagged.append(data.adjectives_tagged)
    else:
        data.json_tagged[-1] = data.adjectives_tagged

    # save
    out_file = open(data.file_path_tagged, "w")
    json.dump(data.json_tagged, out_file)
    out_file.close()


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
            data.relevant  = None
            data.ambiguity = None

    # Continue onto a new item
    elif event.keysym == 'space':
        isExampleCompleteChecks(data)
        if data.isPromptOpen == False:
            save_item(data)
            get_next_item(data)
            data.relevant = None
            data.ambiguity = None
            print("Tagged Adjectives: ", len(data.adjectives_tagged))

    if data.isPromptOpen:
        return

    # If an element is already active, remove it by pressing the key again
    elif event.keysym == data.relevant:
        data.relevant = None
    elif event.keysym == data.ambiguity:
        data.ambiguity = None

    # Add an element to the list of saved items
    elif event.keysym in NUMBERS:
        data.relevant = event.keysym
        print(data.relevant)

    elif event.keysym in ALPHABET:
        data.ambiguity = event.keysym
        print(data.ambiguity)


#########################################################
#########################################################


def drawAdjective(canvas, data):
    adjective_text =  data.adjectives[data.i][0]
    canvas.create_text(10, 10, text='Adjective', font="Arial 18", anchor=NW)
    canvas.create_text(data.width/2, 10, text=adjective_text, font="Arial 20 bold italic", anchor=N)

def drawSentences(canvas, data):

    relevant = 'Is it relevant?'
    if data.relevant != None:
        translation = ""
        if   data.relevant == '0': translation = " (not relevant)"
        elif data.relevant == '1': translation = " (somewhat relevant)"
        elif data.relevant == '2': translation = " (very relevant)"
        info = data.relevant + translation
        relevant += ' ' + info + '\n\n'

    canvas.create_text(10, 100, text=relevant, font="Arial 18 bold", anchor=NW, width= 2 * data.width // 3)

    ambiguity = 'Is it ambiguous?'
    if data.ambiguity != None:
        translation = ""
        if   data.ambiguity == 'a': translation = " (not ambiguous)"
        elif data.ambiguity == 'b': translation = " (somewhat ambiguous)"
        elif data.ambiguity == 'c': translation = " (very ambiguous)"
        info = data.ambiguity + translation
        ambiguity += ' ' + info + '\n\n'

    canvas.create_text(10, 150, text=ambiguity, font="Arial 18 bold", anchor=NW, width=2 * data.width // 3)

def drawDirections(canvas, data):

    canvas.create_text(10, 450, text="Instructions", font="Arial 16 bold", anchor=NW)

    # Context
    context = "Context: " + data.context
    canvas.create_text(10, 40, text=context, font="Arial 14 bold", anchor=NW)

    # Instructions
    inst_relevant = "Type the degree of relevant on a scale from 0 to 2. 0 is not relevant, 2 is very relevant"
    inst_ambiguous = "Type the degree of ambiguity on a scale from a to c. a is not ambiguous, c is very ambiguous"
    canvas.create_text(10, data.height - 80, text=inst_relevant, font="Arial 14", anchor=NW)
    canvas.create_text(10, data.height - 60, text=inst_ambiguous, font="Arial 14", anchor=NW)

    # Criterion
    criterion = """
    The criterion to label the adjectives is as follow:
    Within the given context, try to picture a common object in your head. For instance, if the context is furniture, 
    imagine a chair. If the context is wearable technology, imagine a smart watch. With this image in your mind, if you 
    cannot modify the design of such imaginary picture applying the adjective on the screen, you are most likely to have 
    an ambiguous adjective. The degree of modification should determine the degree of ambiguity. 
    """
    canvas.create_text(10, 480, text="Criterion: ", font="Arial 14 bold", anchor=NW)
    canvas.create_text(5, 500, text=criterion, font="Arial 14", anchor=NW)

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
    drawAdjective(canvas, data)
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

def run(width, height, file_path):
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

    if len(sys.argv) != 2:
        print('*'*20)
        print('Incorrect number of arguments...')
        print('Example: python3 labeling.py ./data/furniture-cleaned.json')
        print('*'*20)


    elif not os.path.exists(sys.argv[-1]):
        print('File %s does not exist.' % sys.argv[-1])

    else:
        run(1080, 720, sys.argv[-1])

