### Cochlear-Project Workflow file 
## Mingi Kang, Abhilasha Kumar 

import argparse 
import os, sys 
import warnings
import zipfile 
import shutil 
import pandas as pd 
import numpy as np 
import matplotlib as mlp

from forager.forager.cues import *
from forager.forager.frequency import * 

# sys.path.append('..')
# from forager.forager.cues import * 
# from forager.forager.frequency import * 
 


"""
Workflow: 
1. Analysize and Organize SFT participant data 
    a. raw-data (original participant data) and replace OOV(out of vocabulary) words
    * 

2. Create different speech2vec and word2vec lexical data combinations with different alphas 
    a. 

"""


#Global variables for dimensions and alpha values 
dimensions = ['50', '100', '200', '300']
type = [
    'alpha_0.0_s2v', # = alpha_1_w2v
    'alpha_0.0_w2v', # = alpha_1_s2v 
    'alpha_0.1_s2v', # = alpha_0.9_w2v
    'alpha_0.1_w2v', # = alpha_0.9_s2v
    'alpha_0.2_s2v', # = alpha_0.8_w2v
    'alpha_0.2_w2v', # = alpha_0.8_s2v
    'alpha_0.3_s2v', # = alpha_0.7_w2v
    'alpha_0.3_w2v', # = alpha_0.7_s2v 
    'alpha_0.4_s2v', # = alpha_0.6_w2v
    'alpha_0.4_w2v', # = alpha_0.6_s2v
    'alpha_0.5_s2v', # = alpha_0.5_w2v
    'alpha_0.5_w2v', 
    'alpha_0.6_s2v', 
    'alpha_0.6_w2v',
    'alpha_0.7_s2v', 
    'alpha_0.7_w2v', 
    'alpha_0.8_s2v', 
    'alpha_0.8_w2v', 
    'alpha_0.9_s2v', 
    'alpha_0.9_w2v', 
    'alpha_1.0_s2v', 
    'alpha_1.0_w2v',
    'average',
    'only_w2v',
    'only_s2v'
]

def initialize_directories(): 
    """
    Initialize directories for the workflow
    """
    # Create directories for the workflow
    
    # fluency_lists folders 
    if not os.path.exists('../forager/data/fluency_lists/participant_results/'):
        os.makedirs('../forager/data/fluency_lists/participant_results/')
    
    # dimension folders 
    for dim in dimensions: 
        
        # lexical data 
        if not os.path.exists('../forager/data/lexical_data/' + dim + '_dim_lexical_data/'): 
            os.makedirs('../forager/data/lexical_data/' + dim + '_dim_lexical_data/')
    
        for t in type: 
            if not os.path.exists('../forager/data/lexical_data/' + dim + '_dim_lexical_data/' + t + '/'):
                os.makedirs('../forager/data/lexical_data/' + dim + '_dim_lexical_data/' + t + '/')
    
        # results
        if not os.path.exists('../forager/output/' + dim + '_dim_results/'):
            os.makedirs('../forager/output/' + dim + '_dim_results/')
    print('Directories created')


def prepare_vocab(data_path): 
    """
    Prepare data for analysis: 
        a. 
        
        
    """
    
    '''Removing Non-Group participants from original data'''
    # Remove the Groupless participant's words (Non NH or CI) 
    data = pd.read_csv(data_path, delimiter='\t')
    
    participant_groupings = pd.read_csv('../forager/data/fluency_lists/participant_data/latest_demo.csv')
    group_dict = dict(zip(participant_groupings['ID'], participant_groupings['Group']))
    no_group_id = []
    
    # Creating participant_groupings.csv for participants' groupings
    participant_groupings_df = pd.DataFrame(columns=['ID', 'Group'])
    participant_id = sorted(list(set(data['ID'].tolist())))
    for id in participant_id:
        if int(id[4:]) in group_dict.keys(): 
            participant_groupings_df.loc[len(participant_groupings_df)] = [id, group_dict[int(id[4:])]]
        else:
            participant_groupings_df.loc[len(participant_groupings_df)] = [id,'n/a']
            no_group_id.append(id)
            
    participant_groupings_df.to_csv('../forager/data/fluency_lists/participant_results/participant_groupings.csv', index=False)
    
    # Creating participant_word_count.csv for number of words produced by participants 
    participant_word_count_df= data['ID'].value_counts().to_frame().sort_values(by=['ID']).reset_index()
    participant_word_count_df.to_csv('../forager/data/fluency_lists/participant_results/participant_word_count.csv', index=False)
    
    # Removed all No ID participants' words from data
    for id in no_group_id: 
        data = data[data['ID']!= id]
    
    
    '''Transforming orignal data with OOV replacements for forager usage'''
    
    # Getting words from Word2Vec and Speech2Vec for OOV words
    s2v_w2v_vocab_list = pd.read_csv("../forager/data/lexical_data/Embeddings/Speech2Vec/speech2vec_50.txt", delimiter=' ', header=0, names=['word'] + list(range(1, 52)))
    s2v_w2v_vocab_list = s2v_w2v_vocab_list['word'].tolist()
    
    # Categorical Animal placements
    animal_categories = pd.read_csv('../forager/data/fluency_lists/participant_data/animals_snafu_scheme.csv')
    
    animal_dict = {} 
    for i in range(len(animal_categories)): 
        if animal_categories['Animal'][i] not in animal_dict: 
            animal_dict[animal_categories['Animal'][i]] = [animal_categories['Category'][i]]
        else:    
            animal_dict[animal_categories['Animal'][i]].append(animal_categories['Category'][i])

    # Variables useful for replacing OOV Words     
    
    OOV_replacements = dict(zip(
        ['gerbil', 'groundhog', 'iguana', 'chimpanzee', 'platypus', 'cheetah', 'hamster', 'emu', 'gecko', 'tyrannosaurusrex', 'dinosaur', 'lemur', 'manatee', 'triceratops', 'waterbug', 'bobcat','wildebeest', 'milkfish', 'anteater', 'armadillo', 'stickbug', 'millipede', 'tarantula', 'dragonfly','stinkbug', 'raccoon', 'stegosaurus', 'orangutan'],
        ['rodents', 'rodents', 'reptile', 'primate', 'water', 'feline', 'pets', 'bird', 'reptile', 'reptile', 'reptile', 'primate', 'water', 'reptile', 'insects', 'feline', 'african','fish', 'insect', 'insect', 'insect', 'insect', 'insect', 'insect','insect', 'american', 'reptile', 'primate']
    ))
    
    
    
    # Creating data-cochlear-replacements.csv for participants' word replacements 
    data_words = data['Word'].tolist() # original 
    
    ## Spell Corrections for original words 
    spelling_correction = dict(zip(['seadragon', 'chimp', 'ring_tailed', 'rhino', 't_rex', 'tigerfish', 'saber_tooth_cat', 'not_snails', 'hippo'], ['seahorse', 'chimpanzee', 'ringtailcat', 'rhinoceros', 'tyrannosaurusrex', 'milkfish', 'sabertoothedtiger', 'snail', 'hippopotamus'] ))
    spell_check_words = [spelling_correction.get(word, word) for word in data['Word'].tolist()] # spell checked 
    
    # words that are in s2v/w2v
    in_s2v_w2v = [] 
    for word in spell_check_words:
        if word not in s2v_w2v_vocab_list: 
            in_s2v_w2v.append('No')
        else:
            in_s2v_w2v.append('Yes')
    
    
    # list of replacements category words from snafu 
    snafu_replacements = [] 
    for word in spell_check_words: 
        if word not in s2v_w2v_vocab_list: 
            snafu_replacements.append(animal_dict[word]) 
        else: 
            snafu_replacements.append("")
            
    # lowercase replacements category words from snafu that are in S2V/W2V
    snafu_replacements_in_s2v_w2v = [] 
    for words in snafu_replacements: 
        if words != "": 
            vocab = [] 
            for word in words: 
                if word == "NorthAmerican": 
                    vocab += ['american']
                if word == "Primates": 
                    vocab += ['primate']
                if word == 'ReptileAmphibian': 
                    vocab += ['reptile']
                    vocab += ['amphibious']
                if word == "Insectivores": 
                    vocab += ['insect']
                if word == "Dinosaur": 
                    vocab += ['reptile']
                if word == "Insects": 
                    vocab += ['insect']
                vocab += [word.lower()]
            snafu_replacements_in_s2v_w2v.append(vocab)
        else: 
            snafu_replacements_in_s2v_w2v.append("")
    
    
    ## Getting OOV words 
    
    final_words = [] 
    for i in range(len(spell_check_words)): 
        if spell_check_words[i] in s2v_w2v_vocab_list: 
            final_words.append(spell_check_words[i])
        else: 
            if len(snafu_replacements_in_s2v_w2v[i]) > 1:
                final_words.append(OOV_replacements[spell_check_words[i]])
            else: 
                final_words.append(snafu_replacements_in_s2v_w2v[i][0])
            
    # CSV table for SFT data OOV replacements 
    data_replacements_df = pd.DataFrame(columns=['ID', 'Word', 'Spell Check', 'In W2V,S2V', 'Replacement Words from Snafu', 'Replacement Words from Snafu in W2V, S2V', 'Final Words'])
    data_replacements_df['ID'] = data['ID'].tolist()
    data_replacements_df['Word'] = data_words
    data_replacements_df['Spell Check'] = spell_check_words
    data_replacements_df['In W2V,S2V'] = in_s2v_w2v
    data_replacements_df['Replacement Words from Snafu'] = snafu_replacements
    data_replacements_df['Replacement Words from Snafu in W2V, S2V'] = snafu_replacements_in_s2v_w2v
    data_replacements_df['Final Words'] = final_words
    data_replacements_df.to_csv('../forager/data/fluency_lists/participant_results/data-cochlear-replacements.csv', index=False)
    
    
    # Creating transformed-data.csv file to later get transformed-data.txt to use for forager input
    transformed_data = pd.DataFrame(columns=['ID', 'Word'])
    transformed_data['Word'] = final_words
    transformed_data['ID'] = data['ID'].tolist()
    transformed_data.to_csv('../forager/data/fluency_lists/participant_data/transformed-data.csv', index=False)


    return None 
def prepare_lexical_data():
    
    # Getting words from Word2Vec and Speech2Vec 
    s2v_w2v_vocab_list = pd.read_csv("../forager/data/lexical_data/Embeddings/Speech2Vec/speech2vec_50.txt", delimiter=' ', header=0, names=['word'] + list(range(1, 52)))
    s2v_w2v_vocab_list = s2v_w2v_vocab_list['word'].tolist()
    
    # participant data 
    participant_data = pd.read_csv('../forager/data/fluency_lists/participant_data/transformed-data.csv')
    participant_words = participant_data['Word'].tolist()
    
    #Snafu data 
    categorical_data = pd.read_csv("../forager/data/fluency_lists/participant_data/animals_snafu_scheme.csv")
    categorical_words = categorical_data['Animal'].tolist()
    
    combined_words = list(set(participant_words + categorical_words))
    combined_words = [x.lower() for x in combined_words]
    combined_words = [x for x in combined_words if x in s2v_w2v_vocab_list]
    
    # print(combined_words)
    
    # had to get the words from the original vocab list due to NLL numbers not matching up 
    combined_words = ['adder', 'albatross', 'alligator', 'alpaca', 'anchovy', 'anemone', 'ant', 'antelope', 'ape', 'asp', 'ass', 'baboon', 'baby', 'badger', 'bantam', 'basilisk', 'bass', 'bat', 'beagle', 'bear', 'beaver', 'bee', 'beetle', 'bird', 'bison', 'bittern', 'blackbird', 'bluebird', 'boa', 'boar', 'booby', 'boxer', 'bruin', 'buck', 'buffalo', 'bug', 'bull', 'bulldog', 'bullfrog', 'bullock', 'bumblebee', 'bunny', 'bunting', 'butterflies', 'butterfly', 'buzzard', 'calf', 'camel', 'cameo', 'canary', 'canine', 'caplin', 'cardinal', 'caribou', 'carp', 'cat', 'caterpillar', 'cattle', 'centipede', 'chaffinch', 'chameleon', 'chamois', 'char', 'chick', 'chickadee', 'chicken', 'chimaera', 'chinook', 'chipmunk', 'clam', 'cobra', 'cochineal', 'cock', 'cockle', 'cod', 'codfish', 'collie', 'colt', 'coney', 'coral', 'cougar', 'courser', 'cow', 'cows', 'coyote', 'crab', 'crane', 'crawfish', 'crayfish', 'cricket', 'crocodile', 'crow', 'cub', 'cuckoo', 'curlew', 'cuttlefish', 'dear', 'deer', 'deerhound', 'devil', 'dodo', 'doe', 'dog', 'dogs', 'dolphin', 'domingo', 'donkey', 'dormouse', 'dory', 'dove', 'dragon', 'drake', 'dromedary', 'duck', 'duckling', 'ducks', 'eagle', 'eel', 'eels', 'eider', 'elephant', 'elk', 'ermine', 'ewe', 'falcon', 'fawn', 'feline', 'ferret', 'filly', 'finch', 'firefly', 'fish', 'fisher', 'flea', 'flies', 'flounder', 'fly', 'flycatcher', 'foal', 'fowl', 'fox', 'foxes', 'foxhound', 'frog', 'fungi', 'gal', 'gamecock', 'gander', 'gar', 'gazelle', 'geese', 'giant', 'gibbon', 'glaucus', 'gnat', 'goat', 'goats', 'goby', 'goldfinch', 'goldfish', 'goose', 'gopher', 'gorilla', 'grasshopper', 'grenadier', 'greyhound', 'grizzly', 'grosbeak', 'grouse', 'grub', 'grunt', 'guanaco', 'guerrilla', 'gull', 'hackney', 'haddock', 'halcyon', 'hare', 'hart', 'hawk', 'hedgehog', 'heifer', 'hen', 'hermit', 'heron', 'herring', 'hippopotamus', 'hog', 'hogs', 'hoopoe', 'horse', 'hound', 'human', 'huntsman', 'husky', 'hydra', 'hyena', 'hyenas', 'ibex', 'insect', 'jack', 'jackal', 'jackass', 'jackdaw', 'jaguar', 'jay', 'jellyfish', 'kangaroo', 'kelpie', 'kid', 'kingfisher', 'kite', 'kitten', 'kitty', 'lab', 'labrador', 'ladybird', 'ladybug', 'lamb', 'lark', 'larva', 'larvae', 'leech', 'lemming', 'leopard', 'linnet', 'lion', 'lioness', 'lions', 'lizard', 'llama', 'loach', 'lobster', 'locust', 'loon', 'louse', 'lynx', 'mackerel', 'maggot', 'magpie', 'maltese', 'mammoth', 'man', 'mare', 'marmot', 'marten', 'mastiff', 'mastodon', 'medusa', 'merlin', 'mice', 'microbe', 'midge', 'mink', 'minnow', 'minx', 'mite', 'moccasin', 'mole', 'mollusk', 'mongoose', 'monitor', 'monkey', 'monkeys', 'monster', 'moose', 'moray', 'mosquito', 'moth', 'moths', 'mouse', 'mule', 'mules', 'mullet', 'muskrat', 'mussel', 'mustang', 'nautilus', 'newt', 'nightingale', 'nuthatch', 'orang', 'oriole', 'ostrich', 'ostriches', 'otter', 'owl', 'ox', 'oxen', 'oyster', 'oysters', 'panther', 'papillon', 'parakeet', 'parrot', 'partridge', 'peacock', 'pelican', 'penguin', 'perch', 'peregrine', 'person', 'pewee', 'pheasant', 'phoenix', 'pig', 'pigeon', 'piggy', 'pike', 'plover', 'polar', 'pollock', 'pony', 'poodle', 'porcupine', 'pork', 'porpoise', 'possum', 'poultry', 'primate', 'ptarmigan', 'puffin', 'pug', 'pullet', 'puma', 'pup', 'puppy', 'puss', 'quail', 'rabbit', 'ram', 'rat', 'rattler', 'rattlesnake', 'raven', 'ray', 'redbird', 'redbreast', 'redtail', 'reindeer', 'reptile', 'rhinoceros', 'roach', 'robin', 'roe', 'rook', 'rooster', 'sable', 'salamander', 'salmon', 'sardine', 'scorpion', 'seal', 'serpent', 'shad', 'shark', 'sheep', 'shellfish', 'shepherd', 'shrew', 'shrimp', 'silkworm', 'skate', 'skunk', 'skylark', 'sloth', 'slug', 'smelt', 'snail', 'snake', 'snapper', 'snipe', 'sow', 'spaniel', 'sparrow', 'spider', 'sponge', 'sprat', 'squab', 'squirrel', 'stag', 'stallion', 'starfish', 'starling', 'steed', 'steer', 'stork', 'sturgeon', 'sunfish', 'swallow', 'swan', 'swift', 'swine', 'tabby', 'tadpole', 'takin', 'tanager', 'tern', 'terrier', 'thoroughbred', 'thrush', 'tick', 'tiger', 'titmouse', 'toad', 'tortoise', 'tortuga', 'triggerfish', 'trotter', 'trout', 'trumpeter', 'tunny', 'turbot', 'turkey', 'turtle', 'unicorn', 'urchin', 'veal', 'vermin', 'viper', 'vulture', 'walrus', 'warbler', 'wasp', 'waterfowl', 'weasel', 'whale', 'whelp', 'whiff', 'whitefish', 'whiting', 'wildcat', 'winkle', 'wolf', 'wolfhound', 'wolverine', 'wolves', 'woman', 'woodchuck', 'woodcock', 'woodpecker', 'worm', 'wren', 'yak', 'yellowlegs', 'zebra', 'girdle', 'oscar', 'pilgrim', 'pegasus', 'guinea', 'lice', 'rainbow', 'killer', 'prairie', 'asian', 'american', 'australian', 'african', 'water', 'rodents', 'insects', 'pets']

    
    # Creating vocab.csv for phonological matrix 
    vocab_df = pd.DataFrame(columns=['Word'])
    vocab_df['Word'] = sorted(list(set(combined_words)))
    vocab_df.to_csv('../forager/data/lexical_data/vocab.csv', index=False)
    
    # Creating Lexical Data (embeddings)
    for dim in dimensions: 
        # reading s2v and w2v text files for embeddings 
        s2v = pd.read_csv("../forager/data/lexical_data/Embeddings/Speech2Vec/speech2vec_" + dim + ".txt", delimiter = " ", header = 1).T
        s2v.drop(s2v.tail(1).index,inplace=True)
        s2v.columns = s2v.iloc[0]
        s2v = s2v[1:]
        
        w2v = pd.read_csv("../forager/data/lexical_data/Embeddings/Word2Vec/word2vec_" + dim + ".txt", delimiter = " ", header = 1).T
        w2v.drop(w2v.tail(1).index,inplace=True)
        w2v.columns = w2v.iloc[0]
        w2v = w2v[1:]
        
        #creating embeddings for only_w2v and only_s2v 
        only_s2v = pd.DataFrame()
        only_w2v = pd.DataFrame()
        
        for word in combined_words: 
            only_s2v[word] = s2v[word].tolist()
            only_w2v[word] = w2v[word].tolist()
        
        only_s2v.to_csv("../forager/data/lexical_data/" + dim + "_dim_lexical_data/only_s2v/embeddings.csv", index=False)
        only_w2v.to_csv("../forager/data/lexical_data/" + dim + "_dim_lexical_data/only_w2v/embeddings.csv", index=False)

        # Creating embeddings for different alpha values 
        average_embeddings = (only_s2v + only_w2v)/2
        average_embeddings.to_csv("../forager/data/lexical_data/" + dim + "_dim_lexical_data/average/embeddings.csv", index=False)
        
        
        for t in type[:22]:
            if t[10:] == 's2v': 
                alp = float(t[6:9])
                s2v_alpha = alp * only_s2v
                w2v_alpha = (1 - alp) * only_w2v 
                
                s2v_alpha = s2v_alpha.reset_index(drop=True) 
                w2v_alpha = w2v_alpha.reset_index(drop=True)
                
                embedding_combined = pd.concat([s2v_alpha, w2v_alpha], ignore_index=True)
                embedding_combined.to_csv("../forager/data/lexical_data/" + dim + "_dim_lexical_data/" + t + "/embeddings.csv", index = False)            
                
            if t[10:] == 'w2v':
                alp = float(t[6:9])
                w2v_alpha = alp * only_w2v
                s2v_alpha = (1 - alp) * only_s2v 
                
                s2v_alpha = s2v_alpha.reset_index(drop=True) 
                w2v_alpha = w2v_alpha.reset_index(drop=True)
                
                embedding_combined = pd.concat([w2v_alpha, s2v_alpha], ignore_index = True)
                embedding_combined.to_csv("../forager/data/lexical_data/" + dim + "_dim_lexical_data/" + t + "/embeddings.csv", index = False)            
                
        
    # Creating semantic matrix 
    for dim in dimensions: 
        for t in type: 
            print("creating sim matrix")

            create_semantic_matrix("../forager/data/lexical_data/" + dim + "_dim_lexical_data/" + t + "/embeddings.csv", "../forager/data/lexical_data/" + dim + "_dim_lexical_data/" + t)
    
    
    # Creating frequencies and Phonological matrix 
    print("creating frequencies")
    get_frequencies("../forager/data/lexical_data/50_dim_lexical_data/only_s2v/embeddings.csv", "../forager/data/lexical_data/50_dim_lexical_data/only_s2v")
                
    print("creating phon matrix") 
    labels, freq_matrix = get_labels_and_frequencies('../forager/data/lexical_data/50_dim_lexical_data/only_s2v/frequencies.csv')
    print(labels)
    phonology_funcs.create_phonological_matrix(labels, '../forager/data/lexical_data/50_dim_lexical_data/only_s2v')
    
    # copying etc lexical data csv files 
    frequency = "../forager/data/lexical_data/50_dim_lexical_data/only_s2v/frequencies.csv"
    phonological_matrix = "../forager/data/lexical_data/50_dim_lexical_data/only_s2v/phonological_matrix.csv"
    vocab = "../forager/data/lexical_data/vocab.csv"

    for dim in dimensions: 
        for t in type:
            try: 
                path = "../forager/data/lexical_data/" + dim + "_dim_lexical_data/" + t
                shutil.copy(frequency, path)
                shutil.copy(phonological_matrix, path)
                shutil.copy(vocab, path)
            except Exception as e: 
                print("")
    
    return None  

def run_workflow(): 

    initialize_directories()
    prepare_vocab('../forager/data/fluency_lists/participant_data/raw-data.txt')
    prepare_lexical_data()
    

# run_workflow()
"After running run_workflow, go to run_foraging_function.py and run the file to get the results"
initialize_directories()