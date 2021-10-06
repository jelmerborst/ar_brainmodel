#nengo
import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask

#other
import numpy as np
import numpy.matlib as matlib

import scipy
import scipy.special
import scipy.sparse

import inspect, os, sys, time, csv, random
#import matplotlib.pyplot as plt
import png ##pypng
import itertools
import base64
import PIL.Image
#import cStringIO
import socket
import warnings
import gc
import datetime


#new components
import spa_mem_voja2RF_pes_hop_twolayers
#import compare_acc
import importlib
importlib.reload(spa_mem_voja2RF_pes_hop_twolayers)
#from voja2_rule import MyOCLsimulator
#import ocl_sim

from ocl_sim import MyOCLsimulator

import cortical_var_route

#new thalamus
import thalamus_var_route
import cortical_var_route

from nengo.utils.builder import default_n_eval_points

#open cl settings:
# 0:0 python == 1 nvidia
# 0:1 python == 3 nvidia
# 0:2 python == 2 nvidia
# 0:3 python == 0 nvidia

if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:3"
else:
    os.environ["PYOPENCL_CTX"] = "0:3"
	


#### SETTINGS #####

nengo_gui_on = __name__ == 'builtins' #python3
ocl = True #use openCL
high_dims = True #True #use full dimensions or not
verbose = True
fixation_time = 200 #ms
fixed_seed = True

print('\nSettings:')

if fixed_seed:
    fseed = datetime.datetime.now()
    fseed = fseed.strftime("%Y%m%d")
    fseed = int(fseed)
    fseed = fseed + 0 #in case we want to change it
    
    fseed=1
    
    np.random.seed(fseed)
    #random.seed(fseed)
    print('\tFixed seed: %i' % fseed)
else:
    fseed = None
    print('\tRandom seed')


if ocl:
	print('\tOpenCL ON')
	import pyopencl
	import nengo_ocl
	ctx = pyopencl.create_some_context()
else:
	print('\tOpenCL OFF')


#set path based on gui
if nengo_gui_on:
    print('\tNengo GUI ON')
    if sys.platform == 'darwin':
        cur_path = '/Users/Jelmer/Work/EM/MEG_fan/models/nengo/assoc_recog'
    elif socket.gethostname() == 'fwn-bborg-5-107':
    	cur_path = '/home/p234584/assoc_recog'
    else:
        cur_path = '/Users/Jelmer/MEG_nengo/assoc_recog'
else:
    print('\tNengo GUI OFF')
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path

print(cur_path)

#set dimensions used by the model
if high_dims:
    D = 512 #768 # 768 #512 #for real words need at least 320, probably move op to 512 for full experiment. Not sure, 96 works fine for half. Maybe 128 for real?
    Dmid = 384 #512 #448 #384 #256 # 256
    Dlow = 128
    print('\tFull dimensions: D = ' + str(D) + ', Dmid = ' + str(Dmid) + ', Dlow = ' + str(Dlow))
else: #lower dims
    D = 384
    Dmid = 192
    Dlow = 96
    print('\tLow dimensions: D = ' + str(D) + ', Dmid = ' + str(Dmid) + ', Dlow = ' + str(Dlow))

print('')


#### HELPER FUNCTIONS ####


#display stimuli in gui, works for 28x90 (two words) and 14x90 (one word)
#t = time, x = vector of pixel values
def display_func(t, x):

    #reshape etc
    if np.size(x) > 14*90:
        input_shape = (1, 28, 90)
    else:
        input_shape = (1,14,90)

    values = x.reshape(input_shape) #back to 2d
    values = values.transpose((1, 2, 0))
    values = (values + 1) / 2 * 255. #0-255
    values = values.astype('uint8') #as ints

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    #make png
    png_rep = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png_rep.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    #html for nengo
    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(img_str))


#load stimuli, subj=0 means a subset of the stims of subject 1 (no long words), works well with lower dims
#short=True does the same, except for any random subject, odd subjects get short words, even subjects long words
word_length = ''
def load_stims(subj=0,short=True,seed=None):
    
    np.random.seed(seed)
    #random.seed(seed)
    
    #subj=0 is old, new version makes subj 0 subj 1 + short, but with a fixed stim set
    sub0 = False
    if subj==0:
        sub0 = True
        subj = 1
        short = True
        longorshort = np.random.randint(2) % 2 == 0

    #set global word length to story correct assoc mem
    global word_length
    

    #pairs and words in experiment for training model
    global target_pairs #learned word pairs
    global target_words #learned words
    
    global target_fan1_pairs #target pairs fan 1 for training
    global target_fan2_pairs #target pairs fan 2 for training

    global items #all items in exp (incl. new foils)
    global rpfoil_pairs #presented rp_foils
    global newfoil_pairs #presented new foils

    #stimuli for running experiment
    global stims #stimuli in experiment
    global stims_target_rpfoils #target re-paired foils stimuli
    global stims_new_foils #new foil stimuli

    #load files (targets/re-paired foils; short new foils; long new foils)
    #ugly, but this way we can use the original stimulus files
    stims = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'Other.txt', skip_header=True,
                          dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                 ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
    stimsNFshort = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFShort.txt', skip_header=True,
                                 dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                        ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
    stimsNFlong = np.genfromtxt(cur_path + '/stims/S' + str(subj) + 'NFLong.txt', skip_header=True,
                                dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                       ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])

    #if short, use a small set of new foils
    if short:
        if sub0: #rand short or long, but not random stims
            if longorshort:
                stimsNF = stimsNFlong[0:8]
                word_length = 'long'
            else:
                stimsNF = stimsNFshort[0:8]
                word_length = 'short'
        elif subj % 2 == 0: #even -> long
            stimsNF = np.random.choice(stimsNFlong,size=8,replace=False) #random.sample(stimsNFlong,8)
            word_length = 'long'
        else: #odd -> short
            stimsNF = np.random.choice(stimsNFshort,size=8,replace=False) #(stimsNFshort,8)
            word_length = 'short'
    else:
        #stimsNF = np.hstack((stimsNFshort,stimsNFlong))
        #change: only pick 8 of each, instead of all
        stimsNF = np.hstack((np.random.choice(stimsNFshort,size=8,replace=False),np.random.choice(stimsNFlong,size=8,replace=False)))
        
    #combine
    stims = np.hstack((stims, stimsNF))
    stims = stims.tolist()

    #if short, only keep shorts for odd subjects or longs for even subjects
    new_stims = []
    if short:
        for i in stims:
            if sub0:
                if longorshort and i[2] == 'Long':
        	        new_stims.append(i)
                elif not(longorshort) and i[2] == 'Short':
                    new_stims.append(i)
            elif subj % 2 == 0 and i[2] == 'Long':
                new_stims.append(i)
            elif subj % 2 == 1 and i[2] == 'Short':
                new_stims.append(i)
        stims = new_stims

    #parse out different categories
    target_pairs = []
    target_fan1_pairs = []
    target_fan2_pairs = []
    rpfoil_pairs = []
    newfoil_pairs = []
    target_words = []
    stims_target_rpfoils = []
    stims_new_foils = []
    items = []

    for i in stims:

        # fill items list with all words
        items.append(i[3])
        items.append(i[4])

        # get target pairs
        if i[0] == 'Target':
            target_pairs.append((i[3],i[4]))
            target_words.append(i[3])
            target_words.append(i[4])
            
            #split fan 1 and fan 2 for training
            if i[1] == 1:
                target_fan1_pairs.append((i[3],i[4]))
            else:
                target_fan2_pairs.append((i[3],i[4]))
            
        elif i[0] == 'RPFoil':
            rpfoil_pairs.append((i[3],i[4]))
            target_words.append(i[3])
            target_words.append(i[4])
        else:
            newfoil_pairs.append((i[3],i[4]))


        # make separate lists for targets/rp foils and new foils (for presenting experiment)
        if i[0] != 'NewFoil':
            stims_target_rpfoils.append(i)
        else:
            stims_new_foils.append(i)

    # remove duplicates
    items = np.unique(items).tolist()
    #items.append('FIXATION')
    target_words = np.unique(target_words).tolist()



# load images for vision
def load_images():

    global X_train, y_train, y_train_words, fixation_image

    indir = cur_path + '/images/'
    files = os.listdir(indir)
    files2 = []

    #select only images for current item set
    for fn in files:
        if fn[-4:] == '.png' and ((fn[:-4] in items)):
             files2.append(fn)

    X_train = np.empty(shape=(np.size(files2), 90*14),dtype='float32') #images x pixels matrix
    y_train_words = [] #words represented in X_train
    for i,fn in enumerate(files2):
            y_train_words.append(fn[:-4]) #add word

            #read in image and convert to 0-1 vector
            r = png.Reader(indir + fn)
            r = r.asDirect()
            
            image_2d = np.vstack(map(np.uint8, r[2]))
           
            image_2d = image_2d / 255
            image_1d = image_2d.reshape(1,90*14)
            X_train[i] = image_1d

    #numeric labels for words (could present multiple different examples of words, would get same label)
    y_train = np.asarray(range(0,len(np.unique(y_train_words))))
    X_train = 2 * X_train - 1  # normalize to -1 to 1


    #add fixation separately (only for presenting, no mapping to concepts)
    r = png.Reader(cur_path + '/images/FIXATION.png')
    r = r.asDirect()
    image_2d = np.vstack(map(np.uint8, r[2]))
    image_2d = image_2d / 255
    fixation_image = np.empty(shape=(1,90*14),dtype='float32')
    fixation_image[0] = image_2d.reshape(1, 90 * 14)

#returns pixels of image representing item (ie METAL)
def get_image(item):
    if item != 'FIXATION':
        return X_train[y_train_words.index(item)]
    else:
        return fixation_image[0]



#### MODEL FUNCTIONS #####


# performs all steps in model ini
def initialize_model(subj=0,seed=None, short=True):

    global subj_gl
    subj_gl = subj
    
    #warn when loading full stim set with low dimensions:
    if not(short) and not(high_dims):
        warn = warnings.warn('Initializing model with full stimulus set, but using low dimensions for vocabs.')

    load_stims(subj,short=short,seed=seed)
    load_images()
    #print(y_train_words)
    initialize_vocabs(seed=seed)


#initialize vocabs
def initialize_vocabs(seed=None):
    
    if seed == None:
        seed = fseed
    
    print('---- INITIALIZING VOCABS ----')

    global vocab_vision #low level visual vocab
    global vocab_concepts #vocab with all concepts and pairs
    global vocab_learned_words #vocab with learned words
    global vocab_all_words #vocab with all words
    global vocab_learned_pairs #vocab with learned pairs
    global vocab_all_pairs #vocab with all pairs, just for decoding
    global vocab_motor #upper motor hierarchy (LEFT, INDEX)
    global vocab_fingers #finger activation (L1, R2)
    global vocab_goal #goal vocab
    global vocab_attend #attention vocab
    global vocab_reset #reset vocab

    global train_targets #vector targets to train X_train on
    global vision_mapping #mapping between visual representations and concepts
    global list_of_pairs #list of pairs in form 'METAL_SPARK'
    global motor_mapping #mapping between higher and lower motor areas
    global motor_mapping_left #mapping between higher and lower motor areas (L1,L2)
    global motor_mapping_right #mapping between higher and lower motor areas (R1,R2)

    #print('vocab seed %g' % seed)

    rng_vocabs = np.random.RandomState(seed=seed)
        
    #low level visual representations
    
    attempts_vocabs = 10000 

    vocab_vision = spa.Vocabulary(Dmid,max_similarity=.05,rng=rng_vocabs) #was .05 .25
    for name in y_train_words:
        vocab_vision.add(name,vocab_vision.create_pointer(attempts=attempts_vocabs))
    train_targets = vocab_vision.vectors


    #word concepts - has all concepts, including new foils, and pairs
    attempts_vocabs = 2000 
    vocab_concepts = spa.Vocabulary(D, max_similarity=.05,rng=rng_vocabs) #was .05
    for i in y_train_words:
        vocab_concepts.add(i,vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('ITEM1',vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('ITEM2',vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('NONE',vocab_concepts.create_pointer(attempts=attempts_vocabs))

    list_of_pairs = []
    list_of_all_pairs = []

    with open('pair_vocab_normed.txt', 'w') as outfile:

        for item1, item2 in target_pairs:
        
        
            #x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
            #vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
            x1 = vocab_concepts.parse('%s*ITEM1' % item1)
            x2 = vocab_concepts.parse('%s*ITEM2' % item2)
            x = x1*(1/x1.length()) + x2*(1/x2.length())
            vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
                
            outfile.write('%s %s ' % (item1, item2))
            vec = (x*(1/x.length())).v
            vec = vec[np.newaxis]
            np.savetxt(outfile, vec, fmt='%.10e')

            #vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
            #    '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
            list_of_pairs.append('%s_%s' % (item1, item2))  # keep list of pairs notation
            list_of_all_pairs.append('%s_%s' % (item1, item2)) 

    # add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        #x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
        #vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
        x1 = vocab_concepts.parse('%s*ITEM1' % item1)
        x2 = vocab_concepts.parse('%s*ITEM2' % item2)
        x = x1*(1/x1.length()) + x2*(1/x2.length())
        vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))

        list_of_all_pairs.append('%s_%s' % (item1, item2)) 

        #vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
        #    '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors
    for item1, item2 in rpfoil_pairs:
        #x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
        #vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
        x1 = vocab_concepts.parse('%s*ITEM1' % item1)
        x2 = vocab_concepts.parse('%s*ITEM2' % item2)
        x = x1*(1/x1.length()) + x2*(1/x2.length())
        vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))      
        
        list_of_all_pairs.append('%s_%s' % (item1, item2)) 

        #vocab_concepts.add('%s_%s' % (item1, item2), vocab_concepts.parse(
        #    '%s*ITEM1 + %s*ITEM2' % (item1, item2)))  # add pairs to concepts to use same vectors

    #print(vocab_concepts['METAL'].v)  

    #vision-concept mapping between vectors
    vision_mapping = np.zeros((D, Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words) # + ['NONE'])

    #vocab with all words
    vocab_all_words = vocab_concepts.create_subset(y_train_words + ['ITEM1', 'ITEM2'])

    #store presented words
    with open('pairs_presented.txt', 'w') as outfile:
        for item1, item2 in target_pairs:
            x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
            outfile.write('target %s %s ' % (item1, item2))
            vec = x.v
            vec = vec[np.newaxis]
            np.savetxt(outfile, vec, fmt='%.10e')        
        for item1, item2 in rpfoil_pairs:
            x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
            outfile.write('rpfoil %s %s ' % (item1, item2))
            vec = x.v
            vec = vec[np.newaxis]
            np.savetxt(outfile, vec, fmt='%.10e')   
            
    #vocab with learned pairs
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs
    
    vocab_all_pairs = vocab_concepts.create_subset(list_of_all_pairs)

    #print vocab_learned_words.keys
    #print y_train_words
    #print target_words

    #motor vocabs, just for sim calcs
    #vocab_motor = spa.Vocabulary(Dmid) #different dimension to be sure, upper motor hierarchy
    #vocab_motor.parse('LEFT+RIGHT+INDEX+MIDDLE')
    vocab_motor = spa.Vocabulary(Dmid,rng=rng_vocabs) #different dimension to be sure, upper motor hierarchy
    vocab_motor.parse('LEFT + RIGHT + INDEX + MIDDLE')
    vocab_motor.add('LEFT_INDEX', vocab_motor.parse('LEFT*INDEX'))
    vocab_motor.add('LEFT_MIDDLE', vocab_motor.parse('LEFT*MIDDLE'))
    vocab_motor.add('RIGHT_INDEX', vocab_motor.parse('RIGHT*INDEX'))
    vocab_motor.add('RIGHT_MIDDLE', vocab_motor.parse('RIGHT*MIDDLE'))
    


    vocab_fingers = spa.Vocabulary(Dlow,rng=rng_vocabs) #direct finger activation
    vocab_fingers.parse('L1 + L2 + R1 + R2')

    #map higher and lower motor
    motor_mapping = np.zeros((Dlow, Dmid))
    motor_mapping += np.outer(vocab_motor.parse('LEFT*INDEX').v, vocab_fingers.parse('L1').v).T
    motor_mapping += np.outer(vocab_motor.parse('LEFT*MIDDLE').v, vocab_fingers.parse('L2').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT*INDEX').v, vocab_fingers.parse('R1').v).T
    motor_mapping += np.outer(vocab_motor.parse('RIGHT*MIDDLE').v, vocab_fingers.parse('R2').v).T
    
       
    #goal vocab
    vocab_goal = spa.Vocabulary(Dmid,rng=rng_vocabs)
    vocab_goal.parse('DO_TASK')
    vocab_goal.parse('ATTEND2')
    vocab_goal.parse('RECOG')
    vocab_goal.parse('RECOG2')
    vocab_goal.parse('FAMILIARITY')
    vocab_goal.parse('START_FAMILIARITY')
    vocab_goal.parse('RECOLLECTION')
    vocab_goal.parse('REPRESENTATION')
    vocab_goal.parse('RESPOND_MATCH')
    vocab_goal.parse('RESPOND_MISMATCH')
    vocab_goal.parse('RESPOND')
    vocab_goal.parse('START_COMPARE_ITEM1')
    vocab_goal.parse('COMPARE_ITEM1')
    vocab_goal.parse('START_COMPARE_ITEM2')
    vocab_goal.parse('COMPARE_ITEM2')
    vocab_goal.parse('END')

    #attend vocab
    vocab_attend = vocab_concepts.create_subset(['ITEM1', 'ITEM2'])

    #reset vocab
    vocab_reset = spa.Vocabulary(Dlow,rng=rng_vocabs)
    vocab_reset.parse('CLEAR + GO')



# word presented in current trial
global cur_item1
global cur_item2
global cur_hand
cur_item1 = 'METAL' #just for ini
cur_item2 = 'SPARK'
cur_hand = 'LEFT'

# returns images of current words for display # fixation for 51 ms.
def present_pair(t):
    if t < (fixation_time/1000.0)+.002:
        return np.hstack((np.ones(7*90),get_image('FIXATION'),np.ones(7*90)))
    else:
        im1 = get_image(cur_item1)
        im2 = get_image(cur_item2)
        return np.hstack((im1, im2))


def present_item2(t, output_attend):

    attn = vocab_attend.dot(output_attend) #dot product with current input (ie ITEM 1 or 2)
    i = np.argmax(attn) #index of max

    attend_scale = attn[i]

    #first fixation before start trial
    if t < (fixation_time/1000.0) + .002:
        # ret_ima = np.zeros(1260)
        ret_ima = attend_scale * get_image('FIXATION')
    else: #then either word or mix of words
        if attend_scale > .1:
            if i == 0: #first item
                ret_ima = attend_scale * get_image(cur_item1)
            else:
                ret_ima = attend_scale * get_image(cur_item2)
        else:
           ret_ima = attend_scale * get_image('FIXATION')
    return ret_ima

#get vector representing hand
def get_hand(t):
    #print(cur_hand)
    return vocab_motor.vectors[vocab_motor.keys.index(cur_hand)]


class AreaIntercepts(nengo.dists.Distribution):
    dimensions = nengo.params.NumberParam('dimensions')
    base = nengo.dists.DistributionParam('base')
    
    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base
        
    def __repr(self):
        return "AreaIntercepts(dimensions=%r, base=%r)" % (self.dimensions, self.base)
    
    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(1-scipy.special.betaincinv((self.dimensions+1)/2.0, 0.5, x+1))
    
    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for i in range(len(s)):
            s[i] = self.transform(s[i])
        return s

#create learning model voja
def create_learning_model_dm_voja(stims, reps, duration, seed=None, n_neurons=200, fan2weights= None):

    if seed == None:
        seed = fseed
    
    np.random.seed(seed) 
        
    to_present = []
    for i in range(reps):
        to_present = to_present + stims
    stims = to_present
 
    #print trial_info
    print('---- INITIALIZING LEARNING MODEL ----')
    global model 
    
    model = spa.SPA(seed=seed)
    with model:

        def stim_func(t): #present stimulus
            if t < duration * len(stims):
                index = int(t / duration)
                scale = 1.0
                
                #words = stims[index].split('_')                
                #alternate_pair = ''
                #for pair in target_pairs:
                #    if pair[0] == words[0] and pair[1] != words[1]: #same start word, other second word
                #        alternate_pair = '%s_%s' % pair
                #alt_words = alternate_pair.split('_')
                #if fan2weights is not None:                    
                #    w = fan2weights[index,:]
                #else:
                #    w = [1,0]

                #if len(alternate_pair) > 0: #fan 2
                     #return w[0] * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v + w[1] * vocab_concepts.parse('%g * %s' % (scale, alternate_pair)).v
                #     return '%g * (%g * (ITEM1*%s + ITEM2*%s) + %g * (ITEM1*%s + ITEM2*%s))' % (scale, w[0], words[0], words[1], w[1], alt_words[0], alt_words[1])
                #else: #fan 1
                    #return 1 * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v
                    #return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
                return '%g * %s' % (scale, stims[index])
            else:        
                return '0'
        #model.stim = nengo.Node(stim_func,size_out=D) #input, can be weaker
             
        model.vis_pair = spa.State(D,vocab=vocab_all_words, label='Visual Pair') 
        #for ens in model.vis_pair.all_ensembles:
        #    ens.neuron_type = nengo.Direct()
        model.input = spa.Input(vis_pair=stim_func)         
                                         
        model.mem = spa_mem_voja2RF_pes_hop_twolayers.SPA_Mem_Voja2RF_Pes_Hop_TwoLayers( ### voja!
                                                input_vocab=vocab_all_words,
                                                n_neurons=n_neurons,
                                                n_neurons_out=20,
                                                dimensions=D,
                                                intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                                intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                                voja_tau=.005, #=default
                                                voja2_rate=1e-1, #.6e-1, #1.5e-2,
                                                voja2_bias=.5, #1, #1 = neg voja, 0 = regular voja.
                                                pes_rate=None,
                                                bcm_rate=None,  
                                                load_from = None,
                                                seed=seed,
                                                ) 
        model.mem.mem.create_weight_probes(duration * len(stims)) #save weights
        
        nengo.Connection(model.vis_pair.output, model.mem.input, synapse=None)
        
        #add some decoding states to see what's going on
        if nengo_gui_on:
            #model.input_retrieval = nengo.Node(None, size_in=D) #D
            #nengo.Connection(model.mem.input, model.input_retrieval, synapse=None)     
            testing_vocab = vocab_all_pairs
            model.stim_state = spa.State(D,vocab= testing_vocab, label='stim state')
            for ens in model.stim_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.stim, model.stim_state.input, synapse=None)
            
        
#create learning model for hopfield
def create_learning_model_dm_hop(stims, reps, duration, seed=None, n_neurons=200, fan2weights= None,load_from=None):

    if seed == None:
        seed = fseed
    
    np.random.seed(seed) 
        
    to_present = []
    for i in range(reps):
        #np.random.shuffle(stims) #don't randomize order here anymore!
        to_present = to_present + stims
    stims = to_present
 
    #print trial_info
    print('---- INITIALIZING LEARNING MODEL ----')
    global model 
    
    model = spa.SPA(seed=seed)
    with model:

        #stims is total list
        #duration is stims + .2 for each stim
        dur_per_stim = duration + 0.2

        def stim_func(t): #present stimulus
                
            mod_t = np.mod(t,dur_per_stim)
            
            if t < dur_per_stim * len(stims): #we present something as long as we're under the total duration
            
                if mod_t <= duration: #stimulus
                    
                    index = int(t / dur_per_stim)
                    scale = .8
                    words = stims[index].split('_')                
                    alternate_pair = ''
                    for pair in target_pairs:
                        if pair[0] == words[0] and pair[1] != words[1]: #same start word, other second word
                            alternate_pair = '%s_%s' % pair
                    alt_words = alternate_pair.split('_')
                    if fan2weights is not None:                    
                        w = fan2weights[index,:]
                    else:
                        w = [1,0]

                    if len(alternate_pair) > 0: #fan 2
                        #return w[0] * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v + w[1] * vocab_concepts.parse('%g * %s' % (scale, alternate_pair)).v
                        #return '%g * (%g * (ITEM1*%s + ITEM2*%s) + %g * (ITEM1*%s + ITEM2*%s))' % (scale, w[0], words[0], words[1], w[1], alt_words[0], alt_words[1])
                        return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
                    else: #fan 1
                        #return 1 * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v
                        return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
                else: #inhibition
                    return '0'
            else:        
                return '0'
                
        def inhib_func(t): #present stimulus
            mod_t = np.mod(t,dur_per_stim)
            
            if t < dur_per_stim * len(stims): #we present something as long as we're under the total duration
                if mod_t <= duration: #stimulus
                    return 0
                else: #inhibition
                    return -1
            else:        
                return 0
                
        #model.stim = nengo.Node(stim_func,size_out=D) #input, can be weaker
        model.vis_pair = spa.State(D,vocab=vocab_all_words, label='Visual Pair') 
        #for ens in model.vis_pair.all_ensembles:
        #    ens.neuron_type = nengo.Direct()
        model.input = spa.Input(vis_pair=stim_func)                               
        
        model.mem = spa_mem_voja2RF_pes_hop_twolayers.SPA_Mem_Voja2RF_Pes_Hop_TwoLayers( ### voja!
                                                input_vocab=vocab_all_words,
                                                n_neurons=n_neurons,
                                                n_neurons_out=20,
                                                dimensions=D,
                                                intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                                intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                                voja2_rate=0, #1.5e-2,
                                                pes_rate=None,
                                                bcm_rate=1e-9,
                                                bcm_theta=1,
                                                bcm_diagonal0=True,
                                                bcm_max_weight=7e-5,  
                                                load_from = load_from,
                                                seed=seed,
                                                ) 
                                               
        model.mem.mem.create_weight_probes(dur_per_stim * len(stims)) #save weights
        nengo.Connection(model.vis_pair.output, model.mem.input, synapse=None)
     
        model.inhib = nengo.Node(inhib_func,size_out=1)
        nengo.Connection(model.inhib, model.mem.mem.mem.neurons,transform=10*np.ones((model.mem.mem.mem.n_neurons,1)))

     
        #add some decoding states to see what's going on
        if nengo_gui_on:
            #model.input_retrieval = nengo.Node(None, size_in=D) #D
            #nengo.Connection(model.mem.input, model.input_retrieval, synapse=None)     
        
            testing_vocab = vocab_all_pairs

            model.stim_state = spa.State(D,vocab= testing_vocab, label='stim state')
            for ens in model.stim_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.stim, model.stim_state.input, synapse=None)

        ### END learning MODEL


#create learning model
def create_learning_model_dm_pes(stims, reps, duration, seed=None, n_neurons=200, fan2weights= None,load_from=None):

    if seed == None:
        seed = fseed
    
    np.random.seed(seed) 
        
    to_present = []
    for i in range(reps):
        #np.random.shuffle(stims) #don't randomize order here anymore!
        to_present = to_present + stims
    stims = to_present

    #print trial_info
    print('---- INITIALIZING LEARNING MODEL ----')
    global model 
    
    model = spa.SPA(seed=seed)
    with model:

        def stim_func(t): #present stimulus
            if t < duration * len(stims):
                index = int(t / duration)
                scale = .8
                
                words = stims[index].split('_')                
                alternate_pair = ''
                for pair in target_pairs:
                    if pair[0] == words[0] and pair[1] != words[1]: #same start word, other second word
                        alternate_pair = '%s_%s' % pair
                alt_words = alternate_pair.split('_')
                if fan2weights is not None:                    
                    w = fan2weights[index,:]
                else:
                    w = [1,0]

                if len(alternate_pair) > 0: #fan 2
                     #return w[0] * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v + w[1] * vocab_concepts.parse('%g * %s' % (scale, alternate_pair)).v
                     return '%g * (%g * (ITEM1*%s + ITEM2*%s) + %g * (ITEM1*%s + ITEM2*%s))' % (scale, w[0], words[0], words[1], w[1], alt_words[0], alt_words[1])
                else: #fan 1
                    #return 1 * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v
                    return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
            else:        
                return '0'
        #model.stim = nengo.Node(stim_func,size_out=D) #input, can be weaker

        model.vis_pair = spa.State(D,vocab=vocab_all_words, label='Visual Pair') 
        #for ens in model.vis_pair.all_ensembles:
        #    ens.neuron_type = nengo.Direct()
        
        def correct_func(t): #correct feedback
            if t < duration * len(stims):
                index = int(t / duration)
                scale = 1.0
                #index = index % len(stims)
                #return vocab_concepts.parse('%g * %s' % (scale, stims[index])).v
                words = stims[index].split('_')                
                alternate_pair = ''
                for pair in target_pairs:
                    if pair[0] == words[0] and pair[1] != words[1]: #same start word, other second word
                        alternate_pair = '%s_%s' % pair
                alt_words = alternate_pair.split('_')
                if fan2weights is not None:                    
                    w = fan2weights[index,:]
                else:
                    w = [1,0]

                if len(alternate_pair) > 0: #fan 2
                     #return w[0] * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v + w[1] * vocab_concepts.parse('%g * %s' % (scale, alternate_pair)).v
                     #return '%g * (%g * (ITEM1*%s + ITEM2*%s) + %g * (ITEM1*%s + ITEM2*%s))' % (scale, w[0], words[0], words[1], w[1], alt_words[0], alt_words[1])
                    return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
                else: #fan 1
                    #return 1 * vocab_concepts.parse('%g * %s' % (scale, stims[index])).v
                    return '%g * (ITEM1*%s + ITEM2*%s)' % (scale, words[0], words[1])
            else:        
                return '0'
                    
        #model.correct = nengo.Node(correct_func,size_out=D) #input correct 
        model.correct = spa.State(D,vocab=vocab_all_words, label='correct') 
        #for ens in model.correct.all_ensembles:
        #    ens.neuron_type = nengo.Direct()
        
        model.input = spa.Input(vis_pair=stim_func, correct=correct_func)    
        
                                         
        model.mem = spa_mem_voja2RF_pes_hop_twolayers.SPA_Mem_Voja2RF_Pes_Hop_TwoLayers( ### voja!
                                                input_vocab=vocab_all_words,
                                                n_neurons=n_neurons,
                                                n_neurons_out=50,
                                                dimensions=D,
                                                intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                                intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                                voja2_rate=0, #1.5e-2,
                                                pes_rate=.03, #.015,
                                                output_radius=10,
                                                bcm_rate=0,  
                                                dec_ini=0,#0.0008, #.0004, #.0003, #.0003,
                                                fwd_multi=1,
                                                load_from = load_from,
                                                seed=seed,
                                                ) 
                                               
        model.mem.mem.create_weight_probes(duration * len(stims)) #save weights
        nengo.Connection(model.vis_pair.output, model.mem.input, synapse=None)
        nengo.Connection(model.correct.output, model.mem.correct, synapse=None)
        
       
        #add some decoding states to see what's going on
        if nengo_gui_on:
            #model.input_retrieval = nengo.Node(None, size_in=D) #D
            #nengo.Connection(model.mem.input, model.input_retrieval, synapse=None)     
        
            testing_vocab = vocab_all_pairs

            model.mem_state = spa.State(D,vocab= testing_vocab, label='mem state')
            for ens in model.mem_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.mem.output, model.mem_state.input, synapse=None)

            model.stim_state = spa.State(D,vocab= testing_vocab, label='stim state')
            for ens in model.stim_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.stim, model.stim_state.input, synapse=None)
            
            model.correct_state = spa.State(D,vocab= testing_vocab, label='correct state')
            for ens in model.correct_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.correct, model.correct_state.input, synapse=None)


        ### END learning MODEL



#run learning model 
#check if assoc mem file exists, if so load, otherwise train mem
def train_associative_memory(seed=None,gui=False,ignore_mem=False,short=True):
    global target_pairs
    global target_fan1_pairs #target pairs fan 1 for training
    global target_fan2_pairs #target pairs fan 2 for training

    np.random.seed(seed) 
    global model

    to_be_learned_pairs = []
    to_be_learned_words = [] #assume voja goes on word stimuli
    
    #first they are all presented once
    tmp = []
    tmp2 = []
    for i in target_pairs:
        tmp.append('%s_%s' % i)
        tmp2.append('%s*ITEM1' % i[0])
        tmp2.append('%s*ITEM2' % i[1])
    np.random.shuffle(tmp)
    to_be_learned_pairs = to_be_learned_pairs + tmp
    np.random.shuffle(tmp2)
    to_be_learned_words = to_be_learned_words + tmp2

    #then we have four learning blocks
    #block 1: fan 1 items 2x, fan 2 items 5x (however, fan 1 items in both direction, so 4x and fan 2 items in 3 pairs (A-B, B-A, C-A) 15x)
    tmp = []
    tmp2 = []
    for i in target_fan1_pairs:
        for j in range(8): #used to be four, but doubled to reflect diff with fan 2 items (see notes 3/4/19)
            tmp.append('%s_%s' % i)
            tmp2.append('%s*ITEM1' % i[0])
            tmp2.append('%s*ITEM2' % i[1])

    for i in target_fan2_pairs:
        for j in range(15):
            tmp.append('%s_%s' % i)
            tmp2.append('%s*ITEM1' % i[0])
            tmp2.append('%s*ITEM2' % i[1])

    np.random.shuffle(tmp)
    to_be_learned_pairs = to_be_learned_pairs + tmp
    np.random.shuffle(tmp2)
    to_be_learned_words = to_be_learned_words + tmp2
    
    #block 2-4, fan 1 pairs 1x, fan 2 items 2x
    for bl in range(3):
        tmp = []
        tmp2 = []
        for i in target_fan1_pairs:
            for j in range(4): #used to be two, but doubled to reflect diff with fan 2 items (see notes 3/4/19)
                tmp.append('%s_%s' % i)
                tmp2.append('%s*ITEM1' % i[0])
                tmp2.append('%s*ITEM2' % i[1])
        for i in target_fan2_pairs:
            for j in range(6):
                tmp.append('%s_%s' % i)
                tmp2.append('%s*ITEM1' % i[0])
                tmp2.append('%s*ITEM2' % i[1])
        np.random.shuffle(tmp)
        to_be_learned_pairs = to_be_learned_pairs + tmp
        np.random.shuffle(tmp2)
        to_be_learned_words = to_be_learned_words + tmp2
    
    #we can use reps to multiply this, that is ok (timing is not correct anyway), ratio will stay the same
    #make model
    reps = 1
    dur = 1 #each presentation. Used to be 2, but we double the amount of presentations for fan 1 items
    
    
    #make fan 2 mixtures.
    tot_reps = len(to_be_learned_pairs)*reps #total nr of presentations

    w = [1.0,0.0] #[1,.3] #25] #indicate the combination of weights #DO NOT USE INTS
    ws = matlib.repmat(w,tot_reps,1)
    
    #if there are random weights, multiply second item of ws with random weights. Otherwise use fixed.
        
    weights2 = None
    #weights2 = np.random.uniform(0,.3,tot_reps) #get random weights for second pair
    
    if weights2 is not None:
        ws[:,1] = weights2
    
    for i in range(len(ws)):
        ws[i,:] = ws[i,:] / np.sum(ws[i,:])            
    
    
    #_voja_hop_pes
    
    ################
    
    #first, train voja
    if short:
        fnamevoja = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
    else:
        fnamevoja = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_subj' + str(subj_gl) + '_full_' + str(seed)

   
    if os.path.isfile(fnamevoja + '.npz') and not ignore_mem and seed != None:
        print('---- VOJA MEMORY ALREADY EXISTS, LOADING ----')
    else: #if no seed or no cache, train associative memory. 
        print('---- LEARNING VOJA ASSOCIATIVE MEMORY ----')
        
        create_learning_model_dm_voja(to_be_learned_words,reps,dur,seed=seed,n_neurons=30000, fan2weights=None) #stims/reps/duration

        #make sim if no nengo_gui or no learning gui
        if not nengo_gui_on or not gui:
            if nengo_gui_on: #make sim in sim
                if ocl:
                    sim = MyOCLsimulator(model,context=ctx,seed=seed)
                else:
                    sim = nengo.simulator.Simulator(model,seed=seed)
            else: #make normal sims

                if ocl:
                    sim = MyOCLsimulator(model,context=ctx,seed=seed,n_prealloc_probes=32)
                else:
                    sim = nengo.Simulator(model,seed=seed)
         
            #run sim
            sim.run(reps*dur*len(to_be_learned_words),progress_bar=verbose)  
    
            #save mem
            model.mem.mem.save(fnamevoja,sim)
    
            #close
            sim.close()
            del sim
            del model
     
    ################
    #for gui check if the previous memory exists
    if os.path.isfile(fnamevoja + '.npz'):

        #second, train hop/BCM
        if short:
            fnamehop = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
        else:
            fnamehop = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_subj' + str(subj_gl) + '_full_' + str(seed)


        if os.path.isfile(fnamehop + '.npz') and not ignore_mem and seed != None:
            print('---- VOJA-HOP MEMORY ALREADY EXISTS, LOADING ----')
        else: #if no seed or no cache, train associative memory. 
            print('---- LEARNING HOP ASSOCIATIVE MEMORY ----')
    
            create_learning_model_dm_hop(to_be_learned_pairs,reps,dur,seed=seed,n_neurons=30000, fan2weights=None,load_from=fnamevoja) #stims/reps/duration

            #make sim if no nengo_gui or no learning gui
            if not nengo_gui_on or not gui:
                if nengo_gui_on: #make sim in sim
                    if ocl:
                        sim = MyOCLsimulator(model,context=ctx,seed=seed,n_prealloc_probes=32)
                    else:
                        sim = nengo.simulator.Simulator(model,seed=seed)
                else: #make normal sims

                    if ocl:
                        sim = MyOCLsimulator(model,context=ctx,seed=seed,n_prealloc_probes=1)
                        #sim = nengo_ocl.Simulator(model,context=ctx,seed=seed,n_prealloc_probes=32)
#prealloc nneurons conn mem
#32 8000 64000k 8513 mb
#32 4000 16000k 2265 mb
#32 20000 400000k  factor 6.25 -> prealloc 32/6 = 5
#32 40000 1600000k factor 25 -> prealloc 1
                    else:
                        sim = nengo.Simulator(model,seed=seed)
     
                #run sim
                sim.run(reps*(dur+.2)*len(to_be_learned_pairs),progress_bar=verbose) #add .2 duration for inhibition 

                #save mem
                model.mem.mem.save(fnamehop,sim)

                #close
                sim.close()
                del sim
                del model         
        
               
    ################

    #for gui check if the previous memory exists
    if os.path.isfile(fnamehop + '.npz'):

        #third, train PES
        if short:
            fnamepes = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
        else:
            fnamepes = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_full_' + str(seed)


        if os.path.isfile(fnamepes + '.npz') and not ignore_mem and seed != None:
            print('---- VOJA-HOP-PES MEMORY ALREADY EXISTS, LOADING ----')
        else: #if no seed or no cache, train associative memory. 
            print('---- LEARNING PES ASSOCIATIVE MEMORY ----')
    
            create_learning_model_dm_pes(to_be_learned_pairs,reps,dur,seed=seed,n_neurons=30000, fan2weights=None,load_from=fnamevoja) #fnamehop #stims/reps/duration

            #make sim if no nengo_gui or no learning gui
            if not nengo_gui_on or not gui:
                if nengo_gui_on: #make sim in sim
                    if ocl:
                        sim = MyOCLsimulator(model,context=ctx,seed=seed)
                    else:
                        sim = nengo.simulator.Simulator(model,seed=seed)
                else: #make normal sims

                    if ocl:
                        sim = MyOCLsimulator(model,context=ctx,seed=seed)
                    else:
                        sim = nengo.Simulator(model,seed=seed)
     
                #run sim
                sim.run(reps*dur*len(to_be_learned_pairs),progress_bar=verbose)  

                #save mem
                model.mem.mem.save(fnamepes,sim)

                #close
                sim.close()
                del sim
                del model         
        
        
           


def create_model(seed=None,short=True):

    global model
    if seed == None:
        seed = fseed

    train_associative_memory(seed=seed,gui=False,ignore_mem=False,short=short)

    if short:
        fname = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
    else:
        fname = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_full_' + str(seed)
        
    ################# REAL MODEL ################
    
    now = datetime.datetime.now()
    
    #print trial_info
    print('---- INITIALIZING MODEL ----')
    
    model = spa.SPA(seed=seed)
    with model:

        model.vis_pair = spa.State(D,vocab=vocab_all_words, feedback=0, feedback_synapse=.05, label='Visual Pair') #was 2, 1.6 
        for ens in model.vis_pair.all_ensembles:
                ens.neuron_type = nengo.Direct()
                

        ##### Recollection & Representation #####
        model.declarative_memory = spa_mem_voja2RF_pes_hop_twolayers.SPA_Mem_Voja2RF_Pes_Hop_TwoLayers(
                                                input_vocab=vocab_all_words,
                                                voja2_rate=0, # no more learning
                                                pes_rate=0, # no more learning
                                                bcm_rate=0, # no more learning
                                                intercepts_out=nengo.dists.Uniform(-1,1),#.1,.1), #=default Uniform(-1.0, 1.0)    
                                                output_radius=10,
                                                fwd_multi=1, #1 
                                                load_from = fname,
                                                seed=seed, 
                                                label = 'DM')
               
        model.input = spa.Input(vis_pair=vispair_func)         


        ### PFC Representation ###
        model.representation = spa.State(D,vocab=vocab_all_words,feedback=1,feedback_synapse=.05,neurons_per_dimension=50, label='Representation')

        rad = 10
        for ens in model.representation.all_ensembles:
                #ens.intercepts=intercepts_out
                ens.radius *= rad
        for c in model.representation.all_connections:
            if c.post_obj is model.representation.output:
                #c.scale_eval_points=False
                ens = c.pre_obj
                n_eval_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
                c.eval_points=ens.eval_points.sample(n_eval_points, ens.dimensions)/rad
    
    
       
        nengo.Connection(model.declarative_memory.output, model.representation.input,transform=1,synapse=.1) #2
     
        #add current forwarding
        model.act_node_mem_out = nengo.Node(None, size_in=1)
        for ens in model.declarative_memory.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_mem_out, transform=np.ones((1, ens.n_neurons))/ens.n_neurons*1, synapse=None)
        
        density = .1 #05
        for ens_out in model.representation.all_ensembles:
            connection_matrix = scipy.sparse.random(ens_out.n_neurons,1,density=density,random_state=seed)
            connection_matrix = connection_matrix != 0
            nengo.Connection(model.act_node_mem_out,ens_out.neurons,transform = connection_matrix.toarray()*.01)


        
          #v2: retrieval done based on comparison input and output
 
        d_comp = D
        model.dm_compare = spa.Compare(d_comp,neurons_per_multiply=250,input_magnitude=.8)
        direct_compare = True
        if direct_compare:
            for ens in model.dm_compare.all_ensembles:
                ens.neuron_type = nengo.Direct()

        nengo.Connection(model.declarative_memory.input[0:d_comp],model.dm_compare.inputA)
        nengo.Connection(model.declarative_memory.output[0:d_comp],model.dm_compare.inputB)


        #output ensemble of comparison
        model.dm_output_ens = nengo.Ensemble(n_neurons=1000,dimensions=1)
        nengo.Connection(model.dm_compare.output,model.dm_output_ens, synapse=None,transform=D/d_comp)

        #feedback
        #nengo.Connection(model.dm_output_ens,model.dm_output_ens,transform=.7,synapse=.1) #.8, .05

        #probe
        model.pr_dm_output_ens = nengo.Probe(model.dm_output_ens,synapse=.01) 
        
        
        #check vector length in mem  layer
        model.vec_len = nengo.Ensemble(500,dimensions=1)
        nengo.Connection(model.declarative_memory.mem.mem,model.vec_len,function=np.linalg.norm)

        #vec len from output directly

        #def sumsq(x):
        #   return np.sum(x**2)

        #model.declarative_memory.mem.output_layer.state_ensembles.add_output('sumsq', sumsq)

        #model.norm2 = nengo.Ensemble(n_neurons=500, dimensions=1)
        #nengo.Connection(model.declarative_memory.mem.output_layer.state_ensembles.sumsq, model.norm2,
        #                transform=np.ones( (1,model.declarative_memory.mem.output_layer.state_ensembles.sumsq.size_out)),synapse=None)

        #model.vec_len = nengo.Ensemble(500,dimensions=1)   
        #nengo.Connection(model.norm2, model.vec_len, function=lambda x: np.sqrt(np.abs(x)))


        model.pr_vec_len = nengo.Probe(model.vec_len,synapse=.01) 


        
        #direct neuron-neuron connection   
        density = .0002
        for ens_out in model.representation.all_ensembles:
            connection_matrix = scipy.sparse.random(ens_out.n_neurons,model.declarative_memory.mem.mem.n_neurons,density=density)
            connection_matrix = connection_matrix != 0
            #nengo.Connection(model.declarative_memory.mem.mem.neurons,ens_out.neurons,transform = connection_matrix.toarray())


        ####### BASAL GANGLIA ######   
        model.bg = spa.BasalGanglia(
            spa.Actions(
            
                start_retrieval =       '1-dot(vis_pair, ITEM1) --> declarative_memory = vis_pair',
                #stop_retrieval =        'retrieval_status_bg --> ',
                z_threshold =           '.5 --> '

            ))
        
        #model.thalamus = spa.Thalamus(model.bg)
        model.thalamus = thalamus_var_route.Thalamus(model.bg, synapse_channel_dict=dict(familiarity=.04,motor=.1,declarative_memory=.04,  representation=.04,comparison_cleanA=.04,comparison_cleanB=.04))

        
        if nengo_gui_on:
            testing_vocab = vocab_all_pairs
            
            #state of memory
            model.mem_state = spa.State(D,vocab=testing_vocab, label='DM State')
            for ens in model.mem_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.declarative_memory.output, model.mem_state.input,synapse=None)
       
            #input
            model.stim_state = spa.State(D,vocab=testing_vocab, label='Stim State')
            for ens in model.stim_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.vis_pair.output, model.stim_state.input, synapse=None)
            
            #representation
            model.rep_state = spa.State(D,vocab=testing_vocab, label='Representation State')
            for ens in model.rep_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.representation.output, model.rep_state.input, synapse=None)
            
        #retrieval activity
        model.act_node_ret = nengo.Node(None, size_in=1)
        nengo.Connection(model.declarative_memory.mem.mem.neurons, model.act_node_ret, transform=np.ones((1, model.declarative_memory.mem.mem.n_neurons)),synapse=None)
        
        model.pr_retrieval = nengo.Probe(model.act_node_ret,synapse=.01) 
        
        #retrieval output activity
        model.act_node_ret_out = nengo.Node(None, size_in=1)
        #nengo.Connection(model.declarative_memory.mem.output_layer.neurons, model.act_node_ret_out, transform=np.ones((1, model.declarative_memory.mem.output_layer.n_neurons)),synapse=None)
        for ens in model.declarative_memory.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_ret_out, transform=np.ones((1, ens.n_neurons)), synapse=None)
        model.pr_retrieval_out = nengo.Probe(model.act_node_ret_out,synapse=.01) 
        
        #retrieval_total
        model.act_node_ret_all = nengo.Node(None, size_in=1)
        nengo.Connection(model.declarative_memory.mem.mem.neurons, model.act_node_ret_all, transform=np.ones((1, model.declarative_memory.mem.mem.n_neurons)),synapse=None)
        #nengo.Connection(model.declarative_memory.mem.output_layer.neurons, model.act_node_ret_all, transform=np.ones((1, model.declarative_memory.mem.output_layer.n_neurons)),synapse=None)
        for ens in model.declarative_memory.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_ret_all, transform=np.ones((1, ens.n_neurons)), synapse=None)
        model.pr_retrieval_all = nengo.Probe(model.act_node_ret_all,synapse=.01) 
        
        #retrieval decoding/confidence
        model.pr_retrieval_decoding = nengo.Probe(model.declarative_memory.mem.output_layer.output,synapse=.01) 
                 
        #representation
        model.representation_activity = nengo.Node(None, size_in=1)
        for ens in model.representation.all_ensembles:
            nengo.Connection(ens.neurons, model.representation_activity, transform=np.ones((1, ens.n_neurons)), synapse=None)
        model.pr_representation = nengo.Probe(model.representation_activity, synapse=0.01) 

        ### END MODEL


stim_detect = 0 ####note: this is not the REAL stim_detect. is in do_trial
def goal_func(t):

    global stim_detect
    if t < (fixation_time/1000.0) + stim_detect:
        return 'WAIT'
    elif t < (fixation_time/1000.0) + stim_detect + .022: #perhaps get from distri
        #print('stim detect ' + str(stim_detect))
        return 'DO_TASK-WAIT'
    elif t < (fixation_time/1000.0) + stim_detect + .082:
    	return '-WAIT'
    else:
        return '0'  # first 50 ms fixation

def vispair_func(t):
    if t > .1 and t < .6:
        return '.8*(ITEM1*%s + ITEM2*%s)' % (cur_item1, cur_item2)
    else:
        return 'ITEM1'
 
#get vector representing hand
def cur_hand_func(t):
    return cur_hand



##### EXPERIMENTAL CONTROL #####

trial_nr = 0
subj_gl = 0
results = []
vision_probe = []
familiarity_probe = []
familiarity_probe2 = []
concepts_probe = []
retrieval_probe = []
retrieval_out_probe = []
retrieval_all_probe = []
representation_probe = []
motor_left_probe = []
motor_right_probe = []

#save results to file
def save_results(fname='output'):
    with open(cur_path + '/data_gpu1/' + fname + '.txt', "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def save_probe(probe,fname='output'):
    with open(cur_path + '/data_gpu1/' + fname + '.txt', "w") as f:
        writer = csv.writer(f)
        writer.writerows(probe)



#prepare simulation
def prepare_sim(seed=None):

    if seed == None:
        seed = fseed

    print('---- BUILDING SIMULATOR ----')

    global sim
    global ctx

    print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')
    #print(vocab_concepts._keys)
    print('\t' + str(len(vocab_concepts.keys)) + ' concepts')

    start = time.time()

    if ocl:
    
    	#   n_prealloc_probes : int (optional)
        #Number of timesteps to buffer when probing. Larger numbers mean less
        #data transfer with the device (faster), but use more device memory. 32 = default
        if nengo_gui_on:
            sim = MyOCLsimulator(model,context=ctx,seed=seed)
        else:
            #print('do we get here?')
            sim = MyOCLsimulator(model,context=ctx,seed=seed)
    else:
        sim = nengo.Simulator(model,seed=seed)
    print('\n ---- DONE in ' + str(round(time.time() - start,2)) + ' seconds ----\n')



total_sim_time = 0

#called by retrieval test
def do_trial_ret(trial_info, hand):

    global total_sim_time
    global results
    global vision_probe
    global familiarity_probe
    global concepts_probe
    global retrieval_probe
    global representation_probe
    global motor_left_probe
    global motor_right_probe
    global familiarity_probe2
    global retrieval_out_probe
    global retrieval_all_probe

    global cur_item1
    global cur_item2
    global cur_hand
    global trial_nr
    global subj
    
    global stim_detect
    stim_detect = .035
    #print('do trial stim_detect: %f' % stim_detect)
    cur_item1 = trial_info[3]
    cur_item2 = trial_info[4]
    cur_hand = hand
    trial_nr += 1

    if verbose:
        print('\n\n---- Trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + ' ----\n')
    
    #run sim 2200 ms, should be enough for responses (incl fixation)
    sim.run(1.001,progress_bar=verbose) #make this shorter than fastest RT

    #judge retrieval
    ret_data = sim.data[model.pr_retrieval_decoding][140:600] #interested in 400-600 ms // now 200-400 as i might it earlier
    #ret_out_data = sim.data[model.pr_retrieval_out_decoding][600:800]
    #print(ret_data.shape) 200x256
    
    ind_vector = spa.similarity(ret_data, vocab_all_pairs)
    #ind_vector_out = spa.similarity(ret_out_data, vocab_all_pairs)

    #print(ind_vector.shape) 200x40 for each of the pairs, we get a match
    
    res = np.argmax(ind_vector, axis=1) #find best SP per time point
    #res = np.argmax(ind_vector_out, axis=1) #NOW USE OUTPUT STATE

    counts = np.bincount(res) #count the winning SPs
    res = np.argmax(counts) #store the one that won
    
    #res_out = np.argmax(ind_vector_out, axis=1) #find best SP per time point
    #counts = np.bincount(res_out) #count the winning SPs
    #res_out = np.argmax(counts) #store the one that won
    
    #print(vocab_all_pairs.keys[res])
    
    conf = np.max(ind_vector) #max match
    #conf_out = np.max(ind_vector_out) #max match

    compare_res = sim.data[model.pr_dm_output_ens]
    compare_res = np.max(compare_res)

    vec_len = sim.data[model.pr_vec_len]
    vec_len = np.max(vec_len)

    #get retrieval time
    resp = -1
    ret_time = -1
    #retacc_data = sim.data[model.pr_retrieval_accumulator]
       
    #print(sim.n_steps)
    #for i in range(int(sim.n_steps)): #range gives 0 to end-1
    
        #calc finger position pr_motor_pos
    #    val = retacc_data[i]
    
    #    if ret_time == -1 and val >= .8: #.8 represents key press
    #        ret_time = i


    #print(conf) #max confidence
    total_sim_time += sim.time
    results.append((subj_gl, trial_nr) + trial_info + (hand, -200, 0, vocab_all_pairs.keys[res], conf, compare_res,ret_time - 100,vec_len))

    #print results

  	#store retrieval
    ret = sim.data[model.pr_retrieval].sum(1)
    retrieval_probe.append([subj_gl,trial_nr] + ret.tolist())
    
    #store retrieval_out
    ret_out = sim.data[model.pr_retrieval_out].sum(1)
    retrieval_out_probe.append([subj_gl,trial_nr] + ret_out.tolist())
    
    #store retrieval_all
    ret_all = sim.data[model.pr_retrieval_all].sum(1)
    retrieval_all_probe.append([subj_gl,trial_nr] + ret_all.tolist())
    

    #store representation
    rep = sim.data[model.pr_representation].sum(1)
    representation_probe.append([subj_gl,trial_nr] + rep.tolist())


def do_retrieval_test(subj=1,short=True,seed=None):

    print('===== RETRIEVAL TEST =====')

    global verbose
    global sim
    verbose = True
    global total_sim_time
    global results
    global trial_nr
    global subj_gl
    global retrieval_probe
    global retrieval_out_probe
    global retrieval_all_probe
    global representation_probe

    
    #subj 0 => subj 1 short
    if subj==0:
    #    subj = 1
        short = True

	
    subj_gl = subj
    trial_nr = 0
    total_sim_time = 0
    results = []
#    concepts_probe = []
    retrieval_probe = []
    retrieval_out_probe = []
    retrieval_all_probe = []
    representation_probe = []

    
    start = time.time()
	
    sim = []
	
    initialize_model(subj=subj,short=short,seed=seed)
#    create_model(seed=seed,short=short)
    train_associative_memory(seed=seed,gui=False,ignore_mem=False,short=short)
    
   #  off when per block:
#     prepare_sim(seed=seed)
# 
# 
#     for each block
#     trial = 0
#     bl = 0
#     
#     clear probes
# 
#    del concepts_probe[:]
#     del retrieval_probe[:]
#     del retrieval_out_probe[:]
#     del retrieval_all_probe[:]
#     del representation_probe[:]
#    del motor_left_probe[:]
#   del motor_right_probe[:]
#    del familiarity_probe2[:]
#         
#     gc.collect()
#     	
#     get all targets/rpfoils for each block
#     stims_in = stims_target_rpfoils
#     add unique new foils if not short
#     if not(short):
#        stims_in = stims_in + nf_short[:8] + nf_long[:8]
#        del nf_short[:8]
#        del nf_long[:8]
#     else: #add fixed nf if short
#        stims_in = stims_in + nf_short + nf_long
# 
#     shuffle
#     np.random.shuffle(stims_in)
# 
#     determine hand
#     if np.random.randint(1,3) == 1:
#         block_hand = 'RIGHT'
#     else:
#         block_hand = 'LEFT'
# 
#     for i in stims_in: #[0:32]: #stims_in:
#         print('\n\nSize ctx = ' + str(sys.getsizeof(ctx)))
#         
#         trial += 1
#         print('Trial ' + str(trial) + '/' + str(len(stims_in)*1))
#         
#         sim.reset()
#         
#         clear out probes
#         for probe in sim.model.probes:
#             del sim._probe_outputs[probe][:]
#         del sim.data
#         sim.data = nengo.simulator.ProbeDict(sim._probe_outputs)
#             
#         do_trial_ret(i, block_hand)
# 
#     save probes/block
#     save_probe(vision_probe,'output_visual_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(familiarity_probe, 'ret_output_familiarity_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(retrieval_probe,'retrieval_output_retrieval_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(retrieval_out_probe,'retrieval_output_retrieval_out_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(retrieval_all_probe,'retrieval_output_retrieval_all_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(representation_probe,'retrieval_output_representation_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(motor_left_probe,'output_left_motor_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(motor_right_probe,'output_right_motor_model_subj' + str(subj) + '_block' + str(bl+1))
#     save_probe(familiarity_probe2, 'output_familiarity2_model_subj' + str(subj) + '_block' + str(bl+1))
# 
#     close and del sim
#     sim.close()
#     del sim
# 
#     print(
#     '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(np.round(total_sim_time,2)) + ' seconds simulation.\n')
# 
#     save behavioral data
#     save_results('retrieval_output_model_subj_' + str(subj))

#if __name__ == '__main__':
    #choice of trial, etc
if not nengo_gui_on:

    #do_1_trial(trial_info=('Target', 1, 'Short', 'METAL', 'SPARK'), hand='RIGHT')
    #do_1_trial(trial_info=('NewFoil', 1, 'Short', 'CARGO', 'HOOD'),hand='LEFT')
    #do_1_trial(trial_info=('RPFoil', 1,	'Short', 'SODA', 'BRAIN'), hand='RIGHT')

    #do_4_trials()

    #do_1_block('RIGHT',subj=1)
    #do_1_block('LEFT')
    #startpp = 1
    #for pp in range(20):
    #    do_experiment(startpp+pp,short=True)
    startpp = 21
    for pp in range(5):
    
        do_retrieval_test(startpp+pp,seed=startpp+pp,short=False)


else:
    #nengo gui on

    #New Foils
    cur_item1 = 'CARGO'
    cur_item2 = 'HOOD'

    #New Foils2
    cur_item1 = 'EXIT'
    cur_item2 = 'BARN'

    #Targets Fan 1 
    #cur_item1 = 'METAL'
    #cur_item2 = 'SPARK'

    #Targets Fan 2
    #cur_item1 = 'FLAME'
    #cur_item2 = 'CAPE'

    #Re-paired foils 1 - Fan 1
    #cur_item1 = 'SODA' 
    #cur_item2 = 'BRAIN'

    #Re-paired foils 2 - Fan 1 
    #cur_item1 = 'METAL' 
    #cur_item2 = 'MOTOR'

    #Re-paired foils 3 - Fan 2
    #cur_item1 = 'FLAME'
    #cur_item2 = 'RACK'

    #Re-paired foils 4 - Fan 2
    #cur_item1 = 'FLAME' 
    #cur_item2 = 'FILE'

    cur_hand = 'LEFT'
    
    seed = 1

    initialize_model(subj=seed,seed=seed,short=True)
    #print(vocab_concepts['METAL'].v)  

    stim_detect = .035 #max([np.random.normal(.042,.011),0]) #was .002 - fixed, easier to compare

    #get random trial:

    stims_in = stims_target_rpfoils
    #stims_in = stims_in + stims_new_foils

    #stims_in=stims_new_foils#

    random.shuffle(stims_in)
    trial_info = stims_in[0]
    cur_item1 = trial_info[3]
    cur_item2 = trial_info[4]
    #determine hand
    if np.random.randint(2) % 2 == 0:
        cur_hand = 'RIGHT'
    else:
        cur_hand = 'LEFT'
    hand = cur_hand

    #train_associative_memory(seed=seed,gui=True,ignore_mem=True)
    create_model(seed=seed,short=True)
 

    print('\n\n---- Trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + ' ----\n')


    print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')






