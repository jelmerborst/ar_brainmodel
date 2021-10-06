#nengo
import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask
from nengo.utils.builder import default_n_eval_points

#other
import numpy as np
import numpy.matlib as matlib

import scipy
import scipy.special
import scipy.sparse

import inspect, os, sys, time, csv, random
import png ##pypng
import itertools
import base64
import PIL.Image
import io #python3
import socket
import warnings
import gc
import datetime

#new components
import assoc_mem_acc
import compare_acc
import spa_mem_voja2_pes_hop_twolayers
from ocl_sim import MyOCLsimulator

#new thalamus
import thalamus_var_route
import cortical_var_route

#open cl settings:
# 0:0 python == 1 nvidia
# 0:1 python == 3 nvidia
# 0:2 python == 2 nvidia
# 0:3 python == 0 nvidia

if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:1"
else:
    os.environ["PYOPENCL_CTX"] = "0:0" #"0:0"


#### SETTINGS #####

nengo_gui_on = __name__ == 'builtins' #python3
ocl = True #use openCL
high_dims = True #True #use full dimensions or not
verbose = True
fixation_time = 200 #ms
fixed_seed = True

print('\nSettings:')

if fixed_seed:
    fseed=1
    np.random.seed(fseed)
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
    else:
        cur_path = '/Users/Jelmer/MEG_nengo/assoc_recog'
else:
    print('\tNengo GUI OFF')
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path
    stim_path = cur_path #'/Users/Jelmer/MEG_nengo/assoc_recog'

#set dimensions used by the model
if high_dims:
    D = 512
    Dmid = 384
    Dlow = 128
    print('\tFull dimensions: D = ' + str(D) + ', Dmid = ' + str(Dmid) + ', Dlow = ' + str(Dlow))
else: #lower dims, only used for quick testing, model will make many mistakes
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
    buffer = io.BytesIO()
    png_rep.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    #html for nengo
    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(str(img_str)))

def BG_func(t):
    #html for nengo
    BG_func._nengo_html_ = '''
           <h4>Basal Ganglia</h4>''' 

def Th_func(t):
    #html for nengo
    Th_func._nengo_html_ = '''
           <h4>Thalamus</h4>''' 


#load stimuli, subj=0 means a subset of the stims of subject 1 (no long words), works  with lower dims
#short=True does the same, except for any random subject, odd subjects get short words, even subjects long words
word_length = ''
def load_stims(subj=0,short=True,seed=None):
    
    np.random.seed(seed)

    #subj=0 is old, new version makes subj 0 subj 1 + short, but with a fixed stim set
    sub0 = False
    if subj==0:
        sub0 = True
        subj = 1
        short = True
        longorshort = np.random.randint(2) % 2 == 0

    #set global word length to store correct assoc mem
    global word_length
    
    #pairs and words in experiment
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
    global stims_rpfoils #only rpfoils
    global stims_targetfan1short
    global stims_fan1
    global stims_fan2
    global stims_targetfan1
    global stims_targetfan2

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
    stims_rpfoils = []
    stims_targetfan1short = []
    stims_fan2 = []
    stims_targetfan2 = []
    stims_fan1 = []
    stims_targetfan1 = []
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
            if i[0] == 'RPFoil':
                stims_rpfoils.append(i)
            else:
                if i[1] == 1 and i[2] == 'Short':
                    stims_targetfan1short.append(i)
                if i[1] == 2:
                    stims_targetfan2.append(i)
                else:
                    stims_targetfan1.append(i)
            if i[1] == 2:
                stims_fan2.append(i)
            else:
                stims_fan1.append(i)
        else:
            stims_new_foils.append(i)

    # remove duplicates
    items = np.unique(items).tolist()
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
            image_2d = np.vstack(list(map(np.uint8, r[2])))
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
    global vocab_all_pairs #vocab with all pairs, only for decoding, not for model itself!
    global vocab_motor #upper motor hierarchy (LEFT, INDEX)
    global vocab_fingers #finger activation (L1, R2)
    global vocab_goal #goal vocab
    global vocab_attend #attention vocab
    global vocab_reset #reset vocab
    global vocab_all_simple_pairs
    
    global train_targets #vector targets to train X_train on
    global vision_mapping #mapping between visual representations and concepts
    global list_of_pairs #list of pairs in form 'METAL_SPARK'
    global motor_mapping #mapping between higher and lower motor areas
    global motor_mapping_left #mapping between higher and lower motor areas (L1,L2)
    global motor_mapping_right #mapping between higher and lower motor areas (R1,R2)

    rng_vocabs = np.random.RandomState(seed=seed)
        
    #low level visual representations
    attempts_vocabs = 10000 
    vocab_vision = spa.Vocabulary(Dmid,max_similarity=.05,rng=rng_vocabs) #was .25
    for name in y_train_words:
        vocab_vision.add(name,vocab_vision.create_pointer(attempts=attempts_vocabs))
    train_targets = vocab_vision.vectors

    #word concepts - has all concepts, including new foils, and pairs
    attempts_vocabs = 2000 
    vocab_concepts = spa.Vocabulary(D, max_similarity=.05,rng=rng_vocabs)
    for i in y_train_words:
        vocab_concepts.add(i,vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('ITEM1',vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('ITEM2',vocab_concepts.create_pointer(attempts=attempts_vocabs))
    vocab_concepts.add('NONE',vocab_concepts.create_pointer(attempts=attempts_vocabs))

    list_of_pairs = []
    list_of_all_pairs = []
    for item1, item2 in target_pairs:
        x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
        vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
        list_of_pairs.append('%s_%s' % (item1, item2))  # keep list of pairs notation
        list_of_all_pairs.append('%s_%s' % (item1, item2)) 

    # add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
        vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
        list_of_all_pairs.append('%s_%s' % (item1, item2)) 

    for item1, item2 in rpfoil_pairs:
        x = vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))
        vocab_concepts.add('%s_%s' % (item1, item2), x*(1/x.length()))
        list_of_all_pairs.append('%s_%s' % (item1, item2)) 
    
    vocab_all_simple_pairs = spa.Vocabulary(D,rng=rng_vocabs)
    
    #add item/word combinations
    list_of_items = []
    for item1, item2 in target_pairs:
    
       #check if it's alreayd in  vocab vocab_concepts['METAL']
        #print(vocab_all_simple_pairs.keys)
        if '%s_ITEM1' % (item1) not in vocab_all_simple_pairs.keys:
            x = vocab_concepts.parse('%s*ITEM1' % (item1))
            vocab_all_simple_pairs.add('%s_ITEM1' % (item1), x*(1/x.length()))
            list_of_items.append('%s_ITEM1' % (item1)) # 
        
        if '%s_ITEM2' % (item2) not in vocab_all_simple_pairs.keys:
            x = vocab_concepts.parse('%s*ITEM2' % (item2))
            vocab_all_simple_pairs.add('%s_ITEM2' % (item2), x*(1/x.length()))
            list_of_items.append('%s_ITEM2' % (item2)) # 

    # add all presented pairs to concepts for display
    for item1, item2 in newfoil_pairs:
        x = vocab_concepts.parse('%s*ITEM1' % (item1))
        vocab_all_simple_pairs.add('%s_ITEM1' % (item1), x*(1/x.length()))
        list_of_items.append('%s_ITEM1' % (item1)) # 
        
        x = vocab_concepts.parse('%s*ITEM2' % (item2))
        vocab_all_simple_pairs.add('%s_ITEM2' % (item2), x*(1/x.length()))
        list_of_items.append('%s_ITEM2' % (item2)) # 
    
    #vision-concept mapping between vectors
    vision_mapping = np.zeros((D, Dmid))
    for word in y_train_words:
        vision_mapping += np.outer(vocab_vision.parse(word).v, vocab_concepts.parse(word).v).T

    #vocab with learned words
    vocab_learned_words = vocab_concepts.create_subset(target_words)

    #vocab with all words
    vocab_all_words = vocab_concepts.create_subset(y_train_words + ['ITEM1', 'ITEM2'])

    #vocab with learned pairs
    vocab_learned_pairs = vocab_concepts.create_subset(list_of_pairs) #get only pairs
    
    vocab_all_pairs = vocab_concepts.create_subset(list_of_all_pairs)

    #motor vocabs
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

# returns images of current words for display 
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
            
#initialize full model
def create_model(seed=None,short=True):

    global model
    global subj_gl
    if seed == None:
        seed = fseed

    if short:
        load_memory_fam = cur_path + '/mem_cache/familiarity_mem_twolayers_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
        load_memory = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_' + word_length + '_' + str(seed)
       
    else:
        load_memory_fam = cur_path + '/mem_cache/familiarity_mem_twolayers_subj' + str(subj_gl) + '_full_' + str(seed)
        load_memory = cur_path + '/mem_cache/assoc_mem_30k_twolayers_voja2RF_hop_pes_subj' + str(subj_gl) + '_full_' + str(seed)

    
    ################# REAL MODEL ################
    
    now = datetime.datetime.now()
    
    
    print('---- INITIALIZING MODEL ----')
    
    model = spa.SPA(seed=seed)
    with model:

        # control
        model.control_net = nengo.Network(seed=seed)
        with model.control_net:
            #assuming the model knows which hand to use (which was blocked)
            model.hand_input = nengo.Node(get_hand)
            model.target_hand = spa.State(Dmid, vocab=vocab_motor, feedback=1)
            nengo.Connection(model.hand_input,model.target_hand.input,synapse=None)

            model.attend = spa.State(D, vocab=vocab_attend, feedback=.6)
            model.goal = spa.State(Dmid, vocab=vocab_goal, feedback=.7)  # current goal
            
            model.dm_time_input = nengo.Node(cur_dm_time_func)
            model.dm_time = spa.State(1,neurons_per_dimension=500)
            nengo.Connection(model.dm_time_input,model.dm_time.input,synapse=None)

        ### vision ###

        #set up network parameters
        n_vis = X_train.shape[1]  # nr of pixels, dimensions of network
        n_hid = 2000 #np.random.randint(1700,2301)  # nr of gabor encoders/neurons.
        print('nhid = %i' % n_hid)
        #store in file
        with open(cur_path + '/data_gpu1/model_settings_subj' + str(subj_gl) + '.txt', "w") as f:
            f.write('subject: %i\n' % subj_gl)
            f.write(now.strftime("%Y-%m-%d %H:%M"))
            f.write('\n--------------------\n\n')
            f.write('n_hid: %i\n' % n_hid)
        
        # random state to start
        rng = np.random.RandomState(seed=seed) 
        encoders = Gabor().generate(n_hid, (9, 9), rng=rng)  # gabor encoders, 7x7
        encoders = Mask((14, 90)).populate(encoders, rng=rng,
                                           flatten=True)  # use them on part of the image

        model.visual_net = nengo.Network(seed=seed)
        with model.visual_net:

            #represent currently attended item
            model.attended_item = nengo.Node(present_item2,size_in=D)
            nengo.Connection(model.attend.output, model.attended_item)

            #vision ensemble
            model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                                                    neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05), 
                                                    encoders=encoders)

            zeros = np.zeros_like(X_train)
            #recurrent connection
            nengo.Connection(model.vision_gabor, model.vision_gabor, synapse=.1,
                             eval_points=np.vstack([X_train, zeros, np.random.randn(*X_train.shape)]),
                             transform=.6)
            #low level visual representation
            model.visual_representation = nengo.Ensemble(n_hid, dimensions=Dmid)

            model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.005, 
                                            eval_points=X_train, function=train_targets,
                                            solver=nengo.solvers.LstsqL2(reg=0.01))
            nengo.Connection(model.attended_item, model.vision_gabor, synapse=.018, transform=3)


        #### Central Cognition ####

        #### Concepts #####
        model.concepts = assoc_mem_acc.AssociativeMemoryAccumulator(
            input_vocab = vocab_all_words,
            wta_output=True, wta_inhibit_scale=1, threshold=.2, #memory params
            status_scale = .5, status_feedback=.2, status_feedback_synapse=0.005,  #status params
                                                                    )
        nengo.Connection(model.visual_representation, model.concepts.input, transform=3*vision_mapping)


        ###### Visual Buffer ######
        model.vis_pair = spa.State(D, vocab=vocab_all_words, feedback= .90, feedback_synapse=.05,neurons_per_dimension=100)
        
        ##### Familiarity #####
        model.familiarity_net = nengo.Network(seed=seed)
        with model.familiarity_net:
            model.familiarity = spa_mem_voja2_pes_hop_twolayers.SPA_Mem_Voja2_Pes_Hop_TwoLayers(
                                                input_vocab=vocab_all_words,
                                                voja2_rate=None, # no voja learning
                                                pes_rate=0, # no more learning
                                                bcm_rate=None,
                                                load_from = load_memory_fam,
                                                intercepts_out=nengo.dists.Uniform(-1,1),
                                                seed=seed,
                                                fwd_dens=.05,
                                                fwd_multi=1,
                                                )
            #cortical connection from vispair to fam - see below
            
           
            #Check for when we have sufficient fam activity to stop input
            model.act_node_fam_input = nengo.Node(None, size_in=1)
            model.act_node_fam_input.output = lambda t, x: x
            #first layer
            nengo.Connection(model.familiarity.mem.mem.neurons, model.act_node_fam_input, transform=np.ones((1, model.familiarity.mem.mem.n_neurons)),synapse=None)
            
            model.fam_done = spa.State(1,feedback=1,neurons_per_dimension=1000, feedback_synapse=.01) 
            
            #switch to maintain status - makes it easier to control
            for ens in model.fam_done.all_ensembles:
               nengo.Connection(ens,ens,function = lambda x: 1 if x > .8 else 0)
      
            # fam accumulator - spa.compare 
            d_comp = D
            model.fam_compare = spa.Compare(d_comp,neurons_per_multiply=250,input_magnitude=.8)
            direct_compare = False #True
            if direct_compare:
                for ens in model.fam_compare.all_ensembles:
                    ens.neuron_type = nengo.Direct()
         
            nengo.Connection(model.familiarity.input[0:d_comp],model.fam_compare.inputA)
            nengo.Connection(model.familiarity.output[0:d_comp],model.fam_compare.inputB)
         
            #output ensemble of comparison
            model.fam_output_ens = nengo.Ensemble(n_neurons=1000,dimensions=1)
            nengo.Connection(model.fam_compare.output,model.fam_output_ens, synapse=None,transform=D/d_comp)
            
            
            thresholdfam = .8 
            def dec_func(x):
                              
                grey = 0.025
                if x > thresholdfam + grey:
                    return 1.0 
                elif x < thresholdfam - grey:
                    return -.08
                else:
                    return x - thresholdfam
                         
            #fam status accumulator
            model.familiarity_status = spa.State(1,neurons_per_dimension=1000,feedback=.9,feedback_synapse=.01) 
            nengo.Connection(model.fam_output_ens, model.familiarity_status.input, 
                 function=dec_func,transform = .8,synapse=.01)
            #switch to maintain status - makes it easier to control
            for ens in model.familiarity_status.all_ensembles:
                nengo.Connection(ens,ens,function = lambda x: 1 if x > .9 else 0)

            #switch to accumulate
            model.do_fam = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.1,wta_output=True)
            nengo.Connection(model.do_fam.am.ensembles[-1], model.familiarity_status.all_ensembles[0].neurons, transform=np.ones((model.familiarity_status.all_ensembles[0].n_neurons, 1)) * -3,
                         synapse=0.02)
            nengo.Connection(model.do_fam.am.ensembles[-1], model.fam_output_ens.neurons, transform=np.ones((model.fam_output_ens.n_neurons, 1)) * -3,
                         synapse=0.01)
        
        
        ##### Recollection & Representation #####
      
        model.declarative_memory_net = nengo.Network(seed=seed)
        with model.declarative_memory_net:

            model.declarative_memory = spa_mem_voja2_pes_hop_twolayers.SPA_Mem_Voja2_Pes_Hop_TwoLayers(
                                                input_vocab=vocab_all_words,
                                                voja2_rate=0, # no more learning
                                                pes_rate=0, # no more learning
                                                bcm_rate=0, # no more learning
                                                intercepts_out=nengo.dists.Uniform(-1,1),    
                                                output_radius=10,
                                                fwd_multi=1,
                                                load_from = load_memory,
                                                seed=seed, 
                                                label = 'DM',
                                                )
                                                
            #cortical connection from vispair to dm - see below
            
            #split above in n_sets
            n_sets = 40
            n_neurons = model.declarative_memory.mem.mem.n_neurons
          
            #use gamma distri with shape 2: m_g = shape*scale_theta
            m_g = .05
            theta = m_g/2
            synapse_distr = np.random.gamma(2, theta, n_sets)
            synapse_distr = np.clip(synapse_distr, 0.005, np.inf)

            set_size = int(n_neurons/n_sets)
            for i in range(n_sets):
                nengo.Connection(model.declarative_memory.mem.mem.neurons[(i*set_size):((i+1)*set_size)], model.declarative_memory.mem.mem.neurons[(i*set_size):((i+1)*set_size)], synapse=synapse_distr[i],seed=seed,transform=.0010*(np.eye(set_size))) # .0010
          
            model.dm_done = spa.State(1,feedback=1,neurons_per_dimension=1000,feedback_synapse=.01) 
            nengo.Connection(model.dm_time_input,model.dm_done.input,synapse=None)

            #switch to maintain status
            for ens in model.dm_done.all_ensembles:
                nengo.Connection(ens,ens,function = lambda x: 1 if x > .75 else 0)
    
            #retrieval based on vector length
            def vec_len_func(x):
                tmp = np.linalg.norm(x)
                return tmp
             
            model.vec_len = nengo.Ensemble(2000,dimensions=2,radius=6)
            nengo.Connection(model.declarative_memory.mem.mem,model.vec_len[0],function=vec_len_func) #np.linalg.norm)
            nengo.Connection(model.vec_len,model.vec_len[1],function=np.max) #max veclen is kept in dim2
          
            
            #rep status accumulator
            model.dm_status = spa.State(1,feedback=1,neurons_per_dimension=2000,feedback_synapse=.01)
            nengo.Connection(model.vec_len[1], model.dm_status.input,transform = .007) 
 
            def dm_func(x):
                 if x > .9:
                    return 1
                 else:
                    return 0
                    
            # switch for easier BG
            model.dm_status2 = spa.State(1,feedback=1,neurons_per_dimension=1000,feedback_synapse=.01)        
            nengo.Connection(model.dm_status.all_ensembles[0],model.dm_status2.input, function=dm_func)
            
            #switch to accumulate
            model.do_dm = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.1,wta_output=True)
            nengo.Connection(model.do_dm.am.ensembles[-1], model.dm_status.all_ensembles[0].neurons, transform=np.ones((model.dm_status.all_ensembles[0].n_neurons, 1)) * -3,
                         synapse=0.02)
                         
            nengo.Connection(model.do_dm.am.ensembles[-1], model.dm_done.all_ensembles[0].neurons, transform=np.ones((model.dm_done.all_ensembles[0].n_neurons, 1)) * -3,
                         synapse=0.02)
          
 		
 		### PFC Representation ###
        model.representation_net = nengo.Network(seed=seed)
        with model.representation_net:
            model.representation = spa.State(D,vocab=vocab_all_words,feedback=1.2,feedback_synapse=.2,neurons_per_dimension=100,subdimensions=1, label='Representation') 

            rad = 10
            for ens in model.representation.all_ensembles:
                    ens.intercepts=nengo.dists.Uniform(0,.1) #-1,1)
                    ens.radius *= rad
            for c in model.representation.all_connections:
                if c.post_obj is model.representation.output:
                    ens = c.pre_obj
                    n_eval_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
                    evpoints =  ens.eval_points.sample(n_eval_points, ens.dimensions)/rad
                    c.eval_points=evpoints
                    
        model.clear_rep = spa.AssociativeMemory(vocab_reset, default_output_key='DO', threshold=.1,wta_output=True)
        for c in model.representation.all_connections:
            if c.post_obj is model.representation.output:
                ens = c.pre_obj
                nengo.Connection(model.clear_rep.am.ensembles[0], ens.neurons, transform=np.ones((ens.n_neurons,1)) * -3, synapse=0.02)
            
            
        #add current forwarding from DM, far below is cortical connection
        model.act_node_mem_out = nengo.Node(None, size_in=1)
        for ens in model.declarative_memory.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_mem_out, transform=np.ones((1, ens.n_neurons))/ens.n_neurons*1, synapse=None)
        
        ffwd_syn_to_rep = .1
        
        density = .8
        for c in model.representation.all_connections:
            if c.post_obj is model.representation.output:
                ens_out = c.pre_obj
                connection_matrix = scipy.sparse.random(ens_out.n_neurons,1,density=density,random_state=seed)
                connection_matrix = connection_matrix != 0
                nengo.Connection(model.act_node_mem_out,ens_out.neurons,transform = connection_matrix.toarray()*.00007,synapse=ffwd_syn_to_rep) #.04) #.0001, .0002
        



        ###### Comparison #####
    
        model.comparison_net = nengo.Network(seed=seed)
        with model.comparison_net:
            d_comp=D
            model.comparison = spa.Compare(d_comp,vocab=vocab_all_words,neurons_per_multiply=250,input_magnitude=.5)
            direct_compare = False #True
            if direct_compare:
                for ens in model.comparison.all_ensembles:
                    ens.neuron_type = nengo.Direct()
        
            #add input cleanup memories
            threshold_cleanup = .1
            wta_inhibit_scale_cleanup = 3
            model.comparison_cleanA = spa.AssociativeMemory(input_vocab=vocab_all_words, wta_output=False, threshold=threshold_cleanup, wta_inhibit_scale=wta_inhibit_scale_cleanup)
            model.comparison_cleanB = spa.AssociativeMemory(input_vocab=vocab_all_words, wta_output=False, threshold=threshold_cleanup, wta_inhibit_scale=wta_inhibit_scale_cleanup)
            nengo.Connection(model.comparison_cleanA.output,model.comparison.inputA)
            nengo.Connection(model.comparison_cleanB.output,model.comparison.inputB)
        
        
            #output ensemble of comparison
            model.comparison_output_ens = nengo.Ensemble(n_neurons=1000,dimensions=1,radius=2)
            nengo.Connection(model.comparison.output,model.comparison_output_ens, synapse=None,transform=D/d_comp)
      
            threshold3 = .9 
            def dec_func3(x):
                grey = 0
                if x > threshold3 + grey:
                    return 1.2
                elif x < threshold3 - grey:
                    return -.15 
                else:
                    return x - threshold3
 
            #rep status accumulator
            model.comparison_status = spa.State(1,neurons_per_dimension=1000,feedback=0.6,feedback_synapse=.01) 
            nengo.Connection(model.comparison_output_ens, model.comparison_status.input,function=dec_func3,transform = 2) 
            
            #do compare
            model.do_comp = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.1,wta_output=True)
            nengo.Connection(model.do_comp.am.ensembles[-1], model.comparison_status.all_ensembles[0].neurons, transform=np.ones((model.comparison_status.all_ensembles[0].n_neurons, 1)) * -3,
                         synapse=0.04)
            nengo.Connection(model.do_comp.am.ensembles[-1], model.comparison_output_ens.neurons, transform=np.ones((model.comparison_output_ens.n_neurons, 1)) * -3,
                         synapse=0.01)
        

        ##### Motor #####
        model.motor_net = nengo.Network(seed=seed)
        with model.motor_net:
			
            #higher motor area 
            model.motor = spa.State(Dmid, vocab=vocab_motor,feedback=0,feedback_synapse=.1)
 			
            #finger area
         
            #split finger areas in left and right (locations of hemi)
            model.fingers_left_hemi = spa.AssociativeMemory(vocab_fingers, input_keys=['R1', 'R2'], wta_output=False,wta_inhibit_scale=1,threshold=0)
            nengo.Connection(model.fingers_left_hemi.output, model.fingers_left_hemi.input, synapse=0.1, transform=0.1) 
             
            model.fingers_right_hemi = spa.AssociativeMemory(vocab_fingers, input_keys=['L1', 'L2'], wta_output=False,wta_inhibit_scale=1,threshold=0)
            nengo.Connection(model.fingers_right_hemi.output, model.fingers_right_hemi.input, synapse=0.1, transform=0.1) 
 
            #connection between higher order area (hand, finger), to lower area
            nengo.Connection(model.motor.output, model.fingers_left_hemi.input, transform=.1*motor_mapping,synapse=.1) 
            nengo.Connection(model.motor.output, model.fingers_right_hemi.input, transform=.1*motor_mapping,synapse=.1)
 
            #finger position
            model.finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4) #order: L1, L2, R1, R2
            nengo.Connection(model.finger_pos.output, model.finger_pos.input, synapse=0.1, transform=0.8)
 
            #record output of pos
            model.finger_pos_state = spa.State(1)
            nengo.Connection(model.finger_pos.output, model.finger_pos_state.input,transform=np.ones((1,4)), synapse=None)
 
            #order: L1, L2, R1, R2
            model.motor_node = nengo.Node(output=motor_node_func, size_in=4,size_out=4,label='motor_node')			
            nengo.Connection(model.fingers_left_hemi.am.elem_output, model.motor_node[2:4])
            nengo.Connection(model.fingers_right_hemi.am.elem_output, model.motor_node[0:2])
            nengo.Connection(model.motor_node, model.finger_pos.input, synapse=None)
			            
         
        ####### BASAL GANGLIA ######   
        motor_multiplier = 1.0 
        vispair_input = 5  
        fam_input = 1.4 
        compare_multi = 4
        dm_input = 0.9  
        rep_input = .1
        motor_background = .4
        model.bg = spa.BasalGanglia(
            spa.Actions(
                #wait & start 
                a_aa_wait =            '1.1*dot(goal,WAIT) - .9 --> goal=0',

                #read & store item1
                a_attend_item1    =    'dot(goal,DO_TASK) - .1 --> goal=1.2*RECOG, attend=1.1*ITEM1', 
                b_store_item1   =      'dot(goal,RECOG) + dot(attend,ITEM1) + concepts_status - .4 --> goal=1.3*ATTEND2, attend=.8*ITEM2, vis_pair=%g*(ITEM1*concepts),  familiarity=%g*(~ITEM1*vis_pair)' % (vispair_input, fam_input), 
                
                #read & store item2
                c_attend_item2  =      'dot(goal,ATTEND2) - 0.0 --> goal=2*RECOG2, attend=0.3*ITEM2', 
                d_store_item2 =        'dot(goal,RECOG2) + concepts_status - .5*fam_done - .2 --> goal=RECOG2+START_FAMILIARITY, vis_pair=%g*(ITEM2*concepts),familiarity=%g*(~ITEM1*vis_pair+~ITEM2*vis_pair),fam_done=.085, do_fam=GO' % (1, fam_input), 
                
                #judge familiarity
                f_accumulate_familiarity =  'dot(goal,START_FAMILIARITY+FAMILIARITY) - dot(goal,RECOLLECTION+COMPARE_ITEM1+COMPARE_ITEM2) + 1.2*fam_done - 1.9--> goal=FAMILIARITY-.4*START_FAMILIARITY, do_fam=GO',
                
                g_respond_unfamiliar = 'dot(goal,FAMILIARITY+RESPOND_MISMATCH_FAM) - dot(goal,RECOLLECTION) - .7*familiarity_status - 1.0 --> goal=RESPOND_MISMATCH_FAM-START_FAMILIARITY-(.5*FAMILIARITY), do_fam=GO, familiarity_status = -1, motor=%g*(target_hand*MIDDLE)' % motor_multiplier, 
                
                h1_start_recollection = 'dot(goal,FAMILIARITY+RECOLLECTION) + 1*familiarity_status - .4 *dm_done - 1.2 --> goal=(RECOLLECTION-START_FAMILIARITY-(.5*FAMILIARITY)), declarative_memory = %g*vis_pair, do_fam=GO, do_dm=GO, familiarity_status = 1' % dm_input,
                                
                #recollection
                h2_recollection =  'dot(goal,RECOLLECTION+START_COMPARE_ITEM1) + dm_done - dm_status2 - 1.7 --> goal=START_COMPARE_ITEM1-.3*RECOLLECTION, do_dm=GO,representation=1.0*declarative_memory', #rule 8

                #comparison - word 1
                j_compare_word1 =    'dot(goal,START_COMPARE_ITEM1+RECOLLECTION+COMPARE_ITEM1) + 1.0*dm_status2 - 1.5 --> goal=COMPARE_ITEM1, comparison_cleanA = %g*(~ITEM1*vis_pair), comparison_cleanB = %g*.1*(~ITEM1*representation),do_comp=GO' % (compare_multi, compare_multi+0), #rule 9
                k1_match_word1 =      'dot(goal,COMPARE_ITEM1) + 1.0*comparison_status - .4 --> goal=1.6*START_COMPARE_ITEM2,motor=%g*(target_hand*INDEX + target_hand*MIDDLE)' % motor_background, #rule 10 
                k2_mismatch_word1 =   'dot(goal,COMPARE_ITEM1) - 1.0*comparison_status - .5 --> goal=1.5*RESPOND_MISMATCH, comparison_status = -1,do_comp=GO, motor=%g*(target_hand*MIDDLE)' % motor_multiplier, #rule 11
                
                #comparison - word 2
                l_compare_word2 =   'dot(goal,START_COMPARE_ITEM2) - .4 --> goal=COMPARE_ITEM2+START_COMPARE_ITEM2-COMPARE_ITEM1, comparison_cleanA = %g*(~ITEM2*vis_pair), comparison_cleanB = %g*.1*(~ITEM2*representation),do_comp=GO,motor=%g*(target_hand*INDEX + target_hand*MIDDLE)' % (compare_multi, compare_multi+0,motor_background), #rule 12
                m1_match_word2 =    'dot(goal,COMPARE_ITEM2) + 1.2*comparison_status - .6 --> goal=1.5*RESPOND_MATCH, motor=%g*(target_hand*INDEX), comparison_status=1,do_comp=GO' % motor_multiplier, #rule 13
                m2_mismatch_word2 = 'dot(goal,COMPARE_ITEM2) - dot(goal, RESPOND_MISMATCH) - 1.2*comparison_status - .6 --> goal=1.5*RESPOND_MISMATCH, comparison_status = -1,do_comp=GO, motor=%g*(target_hand*MIDDLE)' % motor_multiplier, #rule 14
                
                #finish & clean
                z_finished =    '1.0*dot(goal,RESPOND_MISMATCH+RESPOND_MATCH+RESPOND_MISMATCH_FAM) + 1.5*dot(goal,END) + finger_pos_state - 1.8 --> goal=3*END, clear_rep=CLEAR', #-1.7

            ))
        
        model.thalamus = thalamus_var_route.Thalamus(model.bg, synapse_channel_dict=dict(familiarity=.02,motor=.02,declarative_memory=.02,  representation=ffwd_syn_to_rep/2,comparison_A=.02,comparison_B=.02,clear_rep=.1),neurons_channel_dim=25) 

        #add channels for current forwarding to representation
        gate = list()
        channel = list()
        for rule in [8, 9, 10, 12]: 
            gate.append(model.thalamus.get_gate(rule,'representation'))
            with model.representation_net:
                channel.append(nengo.Node(lambda t, x: x[0] if x[1] > -1.5 else 0, size_in=2, label='channel_%d_%s' % (rule, 'currentfwd')))
        
            # inhibit the channel when the action is not chosen
            nengo.Connection(gate[-1], channel[-1][1], transform=-1.5, synapse=.008)
    
            # connect source to target
            nengo.Connection(model.act_node_mem_out, channel[-1][0], synapse=ffwd_syn_to_rep/2)
            density = .8
            for c in model.representation.all_connections:
                if c.post_obj is model.representation.output:
                    ens = c.pre_obj
                    connection_matrix = scipy.sparse.random(ens.n_neurons,1,density=density,random_state=seed)
                    connection_matrix = connection_matrix != 0
                    nengo.Connection(channel[-1] ,ens.neurons,transform = connection_matrix.toarray()*.000015,synapse=ffwd_syn_to_rep/2) 
       
        model.cortical = cortical_var_route.Cortical( # cortical connection: shorthand for doing everything with states and connections
            spa.Actions(
                'familiarity=%g*(~ITEM1*vis_pair+~ITEM2*vis_pair)' % 1.3, 
                'declarative_memory = %g*vis_pair' % 1.5,
                'representation=%g*declarative_memory' % .6, 
         ), synapse_channel_dict=dict(familiarity=.04,declarative_memory=.04,comparison_A=.04,comparison_B=.04,representation=.04)) 

        #probes
        # this one should be on 
        model.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
    
        #fam
        model.act_node_fam = nengo.Node(None, size_in=1)
        #first layer
        nengo.Connection(model.familiarity.mem.mem.neurons, model.act_node_fam, transform=np.ones((1, model.familiarity.mem.mem.n_neurons)),synapse=None)
        #second layer
        for ens in model.familiarity.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_fam, transform=np.ones((1, ens.n_neurons)), synapse=None)
    
        #ret
        model.act_node_ret = nengo.Node(None, size_in=1)
        nengo.Connection(model.declarative_memory.mem.mem.neurons, model.act_node_ret, transform=np.ones((1, model.declarative_memory.mem.mem.n_neurons)),synapse=None)
        for ens in model.declarative_memory.mem.output_layer.all_ensembles:
            nengo.Connection(ens.neurons, model.act_node_ret, transform=np.ones((1, ens.n_neurons)), synapse=None)
    
        #rep
        model.representation_activity = nengo.Node(None, size_in=1)
        for c in model.representation.all_connections:
                if c.post_obj is model.representation.output:
                    ens = c.pre_obj
                    nengo.Connection(ens.neurons, model.representation_activity, transform=np.ones((1, ens.n_neurons)), synapse=None)
    
            
        if True: #not nengo_gui_on:
            
            model.pr_vision_gabor = nengo.Probe(model.vision_gabor.neurons,synapse=.01) 
            model.pr_familiarity = nengo.Probe(model.act_node_fam,synapse=.01)
            model.pr_retrieval = nengo.Probe(model.act_node_ret,synapse=.01)
            model.pr_representation = nengo.Probe(model.representation_activity, synapse=0.01) 
            model.pr_representation_mri = nengo.Probe(model.representation_activity, synapse=0.01)
            model.act_node_motor_left = nengo.Node(None, size_in=model.fingers_left_hemi.am.elem_output.size_out)
            for i, ens in enumerate(model.fingers_left_hemi.am.am_ensembles):
                nengo.Connection(ens.neurons, model.act_node_motor_left[i], transform=np.ones((1, ens.n_neurons)),synapse=None)
            model.pr_motor_left = nengo.Probe(model.act_node_motor_left, synapse=.01)
                        
            model.act_node_motor_right = nengo.Node(None, size_in=model.fingers_right_hemi.am.elem_output.size_out)
            for i, ens in enumerate(model.fingers_right_hemi.am.am_ensembles):
                nengo.Connection(ens.neurons, model.act_node_motor_right[i], transform=np.ones((1, ens.n_neurons)),synapse=None)
            model.pr_motor_right = nengo.Probe(model.act_node_motor_right,synapse=.01)
            model.pr_fam_output_ens = nengo.Probe(model.fam_output_ens,synapse=.01)
            model.pr_comparison_output_ens = nengo.Probe(model.comparison_output_ens,synapse=.01)

        #input
        model.input = spa.Input(goal=goal_func)

        #to show select BG rules
        # get names rules
        if nengo_gui_on:
            vocab_actions = spa.Vocabulary(model.bg.output.size_out)
            for i, action in enumerate(model.bg.actions.actions):
                vocab_actions.add(action.name.upper(), np.eye(model.bg.output.size_out)[i])
            model.actions = spa.State(model.bg.output.size_out,subdimensions=model.bg.output.size_out,
                                  vocab=vocab_actions)
            for ens in model.actions.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.thalamus.output, model.actions.input,synapse=None)

            for net in model.networks:
                if net.label is not None and net.label.startswith('channel'):
                    net.label = ''
            
        #display attended item, only in gui
        if nengo_gui_on:
            model.display_attended = nengo.Node(display_func, size_in=model.attended_item.size_out) 
            nengo.Connection(model.attended_item, model.display_attended, synapse=None)
     
        #show memory + familiarity + rep in GUI                                   
        if nengo_gui_on:
            testing_vocab = vocab_all_pairs
            model.mem_state = spa.State(D,vocab=testing_vocab)
            for ens in model.mem_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.declarative_memory.output, model.mem_state.input,synapse=None)
                    
            testing_vocab = vocab_all_words
            model.fam_state = spa.State(D,vocab=testing_vocab)
            for ens in model.fam_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.familiarity.output, model.fam_state.input,synapse=None)
            
            #representation
            testing_vocab = vocab_all_pairs
            model.rep_state = spa.State(D,vocab=testing_vocab, label='Representation State')
            for ens in model.rep_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.representation.output, model.rep_state.input, synapse=None)
               
            #vis_pair     
            testing_vocab = vocab_all_simple_pairs
            model.vispair_state = spa.State(D,vocab=testing_vocab)
            for ens in model.vispair_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.vis_pair.output, model.vispair_state.input,synapse=None)

        #show comparison
        if nengo_gui_on:
            testing_vocab = vocab_all_words

            model.inputA_state = spa.State(D,vocab=testing_vocab, label='inputA State')
            for ens in model.inputA_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.comparison.inputA, model.inputA_state.input, synapse=None)
            
            model.inputB_state = spa.State(D,vocab=testing_vocab, label='inputB State')
            for ens in model.inputB_state.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(model.comparison.inputB, model.inputB_state.input, synapse=None)

        #display current stimulus pair (not part of model)
        if nengo_gui_on and False:
            model.pair_input = nengo.Node(present_pair)
            model.pair_display = nengo.Node(display_func, size_in=model.pair_input.size_out)  # to show input
            nengo.Connection(model.pair_input, model.pair_display, synapse=None)

		
        ### END MODEL


stim_detect = 0 ####note: this is not the REAL stim_detect. is in do_trial
def goal_func(t):
    global stim_detect
    if t < (fixation_time/1000.0) + stim_detect:
        return 'WAIT'
    elif t < (fixation_time/1000.0) + stim_detect + .022:
        return 'DO_TASK-WAIT'
    elif t < (fixation_time/1000.0) + stim_detect + .082:
    	return '-WAIT'
    else:
        return '0'  

#get vector representing hand
def cur_hand_func(t):
    return cur_hand
    
cur_dm_time = .1 #.05    
def cur_dm_time_func(t):
    global cur_dm_time
    return cur_dm_time

cur_motor_input = 1.2    
def motor_node_func(t,x):
    global cur_motor_input
    return x * cur_motor_input * np.array([0.55, .55, .55, .55]) #np.diag([0.55, .54, .55, .54])


##### EXPERIMENTAL CONTROL #####

trial_nr = 0
subj_gl = 0
results = []
vision_probe = []
familiarity_probe = []
familiarity_probe2 = []
concepts_probe = []
retrieval_probe = []
representation_probe = []
representation_mri_probe = []
motor_left_probe = []
motor_right_probe = []
veclen_probe = []

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
    print('\t' + str(len(vocab_concepts.keys)) + ' concepts')

    start = time.time()

    if ocl:
    
        if nengo_gui_on:
            sim = nengo_ocl.Simulator(model,context=ctx,n_prealloc_probes=2500,seed=seed)
        else:
            sim = nengo_ocl.Simulator(model,context=ctx,n_prealloc_probes=2500,seed=seed)
    else:
        sim = nengo.Simulator(model,seed=seed)
    print('\n ---- DONE in ' + str(round(time.time() - start,2)) + ' seconds ----\n')

total_sim_time = 0

#called by all functions to do a single trial
def do_trial(trial_info, hand):

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
    global representation_mri_probe
    global ret_input_probe
    global veclen_probe

    global cur_item1
    global cur_item2
    global cur_hand
    global trial_nr
    global subj
    
    global stim_detect
    stim_detect = max([np.random.normal(.030,.012),0])
    global ret_bump
    ret_bump = np.random.uniform(.2,.4)

    global cur_dm_time
    cur_dm_time = np.clip(np.random.normal(.06,.02),0.01,None) #.08sd
    
    global cur_motor_input
    cur_motor_input = max([np.random.normal(1.2,.2),.8])

    cur_item1 = trial_info[3]
    cur_item2 = trial_info[4]
    cur_hand = hand
    trial_nr += 1

    if verbose:
        print('\n\n---- Trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + ' ----\n')
    
    sim.run(3.001,progress_bar=verbose) 
    
    #get RT and finger
    resp = -1
    resp_step = -1
    motor_data = sim.data[model.pr_motor_pos]

    for i in range(int(sim.n_steps)): 
    
        motor_pos = motor_data[i]
        position_finger = np.max(motor_pos)

        if resp_step == -1 and position_finger > .7: 
            resp_step = i
            print(motor_pos)  
            resp = np.argmax(motor_pos)

        
    #order: L1, L2, R1, R2
    if verbose:
        if resp == 0:
            print('Left Index')
        elif resp == 1:
            print('Left Middle')
        elif resp == 2:
            print('Right Index')
        elif resp == 3:
            print('Right Middle')
        if resp == -1:
            print('No response')

            
    #response for assoc recog:
    acc = 0 #default 0
    if trial_info[0] == 'Target':
        if (resp == 0 and hand == 'LEFT') or (resp == 2 and hand == 'RIGHT'):
            acc = 1
    else: #new foil & rp foil
        if (resp == 1 and hand == 'LEFT') or (resp == 3 and hand == 'RIGHT'):
            acc = 1

    #fam and comparison max output
    fam = sim.data[model.pr_fam_output_ens]
    fam = np.max(fam)
    comp = sim.data[model.pr_comparison_output_ens]
    comp = np.max(comp)

    if verbose:
        print('RT = ' + str(resp_step-1-200) + ', acc = ' + str(acc))
    total_sim_time += sim.time
    results.append((subj_gl, trial_nr) + trial_info + (hand, (resp_step-1-200), acc, resp, fam, comp))
	
    #store vision
    vis = sim.data[model.pr_vision_gabor].sum(1)
    vis = vis.tolist()
    vision_probe.append([subj_gl,trial_nr] + vis)
    
    #store familiarity
    fam = sim.data[model.pr_familiarity].sum(1)
    fam = fam.tolist()
    familiarity_probe.append([subj_gl, trial_nr] + fam)
     
	#store retrieval
    ret = sim.data[model.pr_retrieval].sum(1)
    retrieval_probe.append([subj_gl,trial_nr] + ret.tolist())
    
    #store representation (= 2200 x 1, but the sum makes it exactly the same as the others)
    rep = sim.data[model.pr_representation].sum(1)
    representation_probe.append([subj_gl,trial_nr] + rep.tolist())
    
    #store representation fMRI
    repmri = sim.data[model.pr_representation_mri].sum(1)
    representation_mri_probe.append([subj_gl,trial_nr] + repmri.tolist())
    
    #store motor left
    ml = sim.data[model.pr_motor_left].sum(1)
    motor_left_probe.append([subj_gl,trial_nr] + ml.tolist())
    
    #store motor right
    mr = sim.data[model.pr_motor_right].sum(1)
    motor_right_probe.append([subj_gl,trial_nr] + mr.tolist())

 
def do_experiment(subj=1,short=True,seed=None):

    print('===== RUNNING FULL EXPERIMENT =====')

    #mix of MEG and EEG experiment
    #14 blocks (7 left, 7 right)
    #64 targets/rp-foils per block + 16 new foils (EEG)
    #if short == True, 32 targets/rp-foils + 8 new foils (only short (odd subjects) or long words (even subjects))
    #for full exp total number new foils = 14*16=224. We only have 208, but we can repeat some.
    #for short exp we repeat a set of 8 foils each block (model is reset anyway)
    global verbose
    global sim
    verbose = True
    global total_sim_time
    global results
    global trial_nr
    global subj_gl
    global vision_probe
    global familiarity_probe
    global concepts_probe
    global retrieval_probe
    global representation_probe
    global motor_left_probe
    global motor_right_probe
    global familiarity_probe2
    global representation_mri_probe
    global ret_input_probe
    global veclen_probe
    
    #subj 0 => subj 1 short
    if subj==0:
    #    subj = 1
        short = True

	
    subj_gl = subj
    trial_nr = 0
    total_sim_time = 0
    results = []
    vision_probe = []
    familiarity_probe = []
    concepts_probe = []
    retrieval_probe = []
    representation_probe = []
    motor_left_probe = []
    motor_right_probe = []
    familiarity_probe2 = []
    representation_mri_probe = []
    ret_input_probe = []
    veclen_probe = []
    
    start = time.time()
	
    sim = []

    initialize_model(subj=subj,short=short,seed=seed)
    create_model(seed=seed,short=short)
    
    prepare_sim(seed=seed)

    #for each block
    trial = 0
    for bl in range(14):
		
        print('Block ' + str(bl+1) + '/14')
		
        #clear probes
        del vision_probe[:]
        del familiarity_probe[:]
        del concepts_probe[:]
        del retrieval_probe[:]
        del representation_probe[:]
        del motor_left_probe[:]
        del motor_right_probe[:]
        del familiarity_probe2[:]
        del representation_mri_probe[:]
        del ret_input_probe[:]
        del veclen_probe[:]
       
        gc.collect()
    	
        # get all targets/rpfoils for each block
        stims_in = stims_target_rpfoils
        stims_in = stims_in + stims_new_foils 

        #shuffle
        np.random.shuffle(stims_in)

        #determine hand
        if (bl+subj) % 2 == 0:
            block_hand = 'RIGHT'
        else:
            block_hand = 'LEFT'

        for i in stims_in: #stims_in:
        	
            trial += 1
            print('Trial ' + str(trial) + '/' + str(len(stims_in)*14))
            
            sim.reset()
            
			# clear out probes
            for probe in sim.model.probes:
                del sim._probe_outputs[probe][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs)
    			
            do_trial(i, block_hand)
    
        #save probes/block
        save_probe(vision_probe,'output_visual_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(familiarity_probe, 'output_familiarity_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(retrieval_probe,'output_retrieval_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(representation_probe,'output_representation_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(representation_mri_probe,'output_representation_mri_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(motor_left_probe,'output_left_motor_model_subj' + str(subj) + '_block' + str(bl+1))
        save_probe(motor_right_probe,'output_right_motor_model_subj' + str(subj) + '_block' + str(bl+1))

    #close and del sim
    sim.close()
    del sim

    print(
    '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')

    # save behavioral data
    save_results('output_model_subj_' + str(subj))



def do_exp_test(subj=1,short=True,seed=None): # 1 block

    print('===== RETRIEVAL TEST =====')
    print(seed)
    global verbose
    global sim
    verbose = True
    global total_sim_time
    global results
    global trial_nr
    global subj_gl
    global vision_probe
    global familiarity_probe
    global concepts_probe
    global retrieval_probe
    global representation_probe
    global motor_left_probe
    global motor_right_probe
    global familiarity_probe2
    global representation_mri_probe
    global veclen_probe
    
    #subj 0 => subj 1 short
    if subj==0:
    #    subj = 1
        short = True

	
    subj_gl = subj
    trial_nr = 0
    total_sim_time = 0
    results = []
    vision_probe = []
    familiarity_probe = []
    retrieval_probe = []
    representation_probe = []
    motor_left_probe = []
    motor_right_probe = []
    representation_mri_probe = []
    veclen_probe = []
    
    start = time.time()
	
    sim = []
	
    initialize_model(subj=subj,short=short,seed=seed)
    create_model(seed=seed,short=short)
    
    #off when per block:
    prepare_sim(seed=seed)

    #for each block
    trial = 0
    bl = 0
    
    #clear probes
    del vision_probe[:]
    del familiarity_probe[:]
    del retrieval_probe[:]
    del representation_probe[:]
    del motor_left_probe[:]
    del motor_right_probe[:]
    del representation_mri_probe[:]
    del veclen_probe[:]
        
    gc.collect()
    	
    # get all targets/rpfoils for each block
    stims_in = stims_target_rpfoils

    #add unique new foils if not short
    stims_in = stims_in + stims_new_foils

    #shuffle
    np.random.shuffle(stims_in)

    #determine hand
    if np.random.randint(1,3) == 1:
        block_hand = 'RIGHT'
    else:
        block_hand = 'LEFT'

    for i in stims_in: #stims_in:
        
        trial += 1
        print('Trial ' + str(trial) + '/' + str(len(stims_in)*1))
        
        sim.reset()
        
        # clear out probes
        for probe in sim.model.probes:
            del sim._probe_outputs[probe][:]
        del sim.data
        sim.data = nengo.simulator.ProbeDict(sim._probe_outputs)
            
        do_trial(i, block_hand)

    #save probes/block
    save_probe(vision_probe,'ret_output_visual_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(familiarity_probe, 'ret_output_familiarity_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(retrieval_probe,'ret_output_retrieval_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(representation_probe,'ret_output_representation_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(representation_mri_probe,'ret_output_representation_mri_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(motor_left_probe,'ret_output_left_motor_model_subj' + str(subj) + '_block' + str(bl+1))
    save_probe(motor_right_probe,'ret_output_right_motor_model_subj' + str(subj) + '_block' + str(bl+1))

    #close and del sim
    sim.close()
    del sim

    print(
    '\nTotal time: ' + str(round(time.time() - start, 2)) + ' seconds for ' + str(round(total_sim_time,2)) + ' seconds simulation.\n')

    # save behavioral data
    save_results('ret_output_model_subj_' + str(subj))


#choice of trial, etc
startpp_both = 1
if not nengo_gui_on:
 
    startpp = startpp_both
    
    for pp in range(1):
        do_experiment(startpp+pp,short=False,seed=startpp+pp)
        #do_exp_test(startpp+pp,seed=startpp+pp,short=False)
else:
    #nengo gui on
    
    initialize_model(subj=startpp_both,seed=startpp_both,short=False)
    create_model(seed=startpp_both,short=False)

    stim_detect = max([np.random.normal(.030,.012),0])
    ret_bump = np.random.uniform(.2,.4)
    cur_dm_time = np.clip(np.random.normal(.06,.02),0.01,None) #.08sd
    cur_motor_input = max([np.random.normal(1.2,.2),.8])

	#get random trial:
    stims_in = stims_target_rpfoils
    
    random.shuffle(stims_in)
    trial_info = stims_in[0]
    
    #determine hand
    if np.random.randint(2) % 2 == 0:
        cur_hand = 'RIGHT'
    else:
        cur_hand = 'LEFT'
    hand = cur_hand
    
    print(trial_info)
   
    cur_item1 = trial_info[3]
    cur_item2 = trial_info[4]
	
    print('\n\n---- Trial: ' + trial_info[0] + ', Fan ' + str(trial_info[1])
          + ', ' + trial_info[2] + ' - ' + ' '.join(trial_info[3:]) + ' - ' + hand + ' ----\n')

    print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')
