import numpy as np 
import pandas as pd 
import nltk
import scipy
from functools import lru_cache
from itertools import product as iterprod
import re
from tqdm import tqdm
import os


# write funcs that check lexical data format for errors

class check_lexical:
    def check_lexical_files(dim_folder_path):
        mean_sim_df = pd.DataFrame(columns=["alpha", "model", "mean_sem", "mean_phon"])

        for root, dirs, files in os.walk(dim_folder_path):
            print("in folder: ", root)
            print("in directory: ", dirs)
            print("in files: ", files)

            if len(files)!= 0:

                phon = pd.read_csv(root  + "/phonological_matrix.csv", header=None)
                sem = pd.read_csv(root  + "/semantic_matrix.csv", header=None)
                freq = pd.read_csv(root  + "/frequencies.csv", header=None)
                vocab = pd.read_csv(root  +  "/vocab.csv", header=0)
                embeddings = pd.read_csv(root  + "/embeddings.csv", header=0)

                # verify dimensions
                
                assert phon.shape[0] == sem.shape[0] == freq.shape[0] == vocab.shape[0] == embeddings.shape[1], "dimension mismatch"

                # verify that the same words are in the vocab and the embeddings and frequencies
                assert vocab["Word"].tolist() == freq[0].tolist(), "vocab and freq don't match"
                assert vocab["Word"].tolist() == embeddings.columns.to_list(), "vocab and embeddings don't match"

                # verify that matrices do not contain values above 1 or below -1
                assert (phon > 1).sum().sum() == 0, "phonological matrix contains values above 1"
                assert (phon < -1).sum().sum() == 0, "phonological matrix contains values below -1"
                assert (sem > 1).sum().sum() == 0, "semantic matrix contains values above 1"
                assert (sem < -1).sum().sum() == 0, "semantic matrix contains values below -1"

                # verify that the semantic matrix for alphas2v is equivalent to the semantic matrix for (1-alpha)w2v


                # root should be of the form ../forager/data/lexical_data/50_dim_lexical_data/alpha_1.0_s2v
                # get last part of the folder name
                folder_key_parts = root.split("/")[-1].split("_") 
                if(len(folder_key_parts) == 3): # not checking for average models or only_w2v or only_s2v
                    alpha = round(float(folder_key_parts[1]),1)
                    one_minus_alpha = round(1 - alpha, 1)
                    model = folder_key_parts[2]
                    othermodel = "s2v" if model == "w2v" else "w2v"

                    print("checking semantic matrices for alpha: ", alpha, "and ", one_minus_alpha, " and models: ", othermodel, " and ", model)

                    # read in the 1-alpha matrix 
                    one_minus_alpha_sem = pd.read_csv(dim_folder_path + "alpha_" + str(one_minus_alpha) + "_" + othermodel + "/semantic_matrix.csv", header=None)

                    # check that the matrices are equivalent
                    # first round all values to 2 decimal places
                    sem = sem.round(2)
                    one_minus_alpha_sem = one_minus_alpha_sem.round(2)

                    assert (sem == one_minus_alpha_sem).all().all(), "semantic matrices are not equivalent for alpha: " + str(alpha) + ": " + model+ " and "+ str(one_minus_alpha) + ": " + othermodel
                
                    # get mean semantic and mean phonological similarity
                    mean_sem = sem.mean().mean()
                    mean_phon = phon.mean().mean()
                    # add them to a new dataframe
                    mean_sim_df = pd.concat([mean_sim_df, pd.DataFrame({"alpha": alpha, "model": model, "mean_sem": mean_sem, "mean_phon": mean_phon}, index=[0])])
        
        # after for loop, write the mean_sim_df to a csv
        mean_sim_df.to_csv("../forager/data/lexical_data/mean_similarities.csv", index=False)


    
    def check_participant_data(participant_ID):
        # check how many words are in the participant data
        
        # read in all fluency data
        fluency = pd.read_csv("../forager/data/fluency_lists/participant_data/raw-data.txt", header=0, sep="\t")
        print(fluency.head())
        fluency_sub = fluency[fluency["ID"] == participant_ID]
        words = fluency_sub["Word"].tolist()
        # check which words are not in vocab
        vocab = pd.read_csv("../forager/data/lexical_data/FINAL_vocab.csv", header=0)
        vocab_words = vocab["Word"].tolist()
        not_in_vocab = [word for word in words if word not in vocab_words]
        print("words not in vocab: ", not_in_vocab)
        

check_lexical.check_lexical_files("../forager/data/lexical_data/50_dim_lexical_data/")
#check_lexical.check_participant_data('CBM-702')
            