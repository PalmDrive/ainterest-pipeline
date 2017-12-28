import datetime
import pickle
## python script from annotating passages.
import os, json
from IPython.display import clear_output
try: os.mkdir('./datafiles/biaozhu/')
except: pass
def getContentLabelPairs():
    

    def f(x, other=''):
        return {'1' : 'ad',
                '2' : 'non-ad'}.get(x, other)

    content_label_pairs = dict()
    try:
        with open('./log.pickle', 'rb') as f:
            line = pickle.load(f)
    except: 
        line = int(input("the line when you left(press 0 if you want to start from the beginning)\n"))
    with open('./passage.txt') as file:
        content = file.read().split('\n\n')
        content = content[line+1:]    

    for num in range(len(content)):
        passage = content[num]
        print(passage+'\n\n')
        arg = input("""Press 1 for 'ad'\nPress 2 for 'non-ad'\ndirectly enter other\nPress o to output the file\nPress d to skip to the next one\n""")
        # get input from keyboard
        if arg == 'd':
            os.system('clear')
            continue
        elif arg=='o':
            position = line + num
            with open('log.pickle', 'wb') as f:
                pickle.dump(position, f)
            return content_label_pairs
        elif arg == '1':
            content_label_pairs[passage] = 'ad'
        elif arg == '2': 
            content_label_pairs[passage] = 'non-ad'
        else:
            content_label_pairs[passage] = arg
        os.system('clear')
filename = 'test1.json'        
fileIsValid = os.path.exists(filename)
if fileIsValid:
    append_write = 'w' # append if already exists
    append_read = 'r'
else:
    append_write = 'w' # make a new file if not
if __name__ == '__main__':
    
    content = getContentLabelPairs()
    if not fileIsValid:
        with open('test1.json', append_write) as f:           
            json.dump(content, f)
    else:
        with open(filename, append_read) as f:
            lst = json.load(f)         
        z = {**lst, **content}
        print('Number of Passages is {0}'.format(len(list(z.values()))))
        with open(filename, append_write) as f:
            json.dump(z, f)
            