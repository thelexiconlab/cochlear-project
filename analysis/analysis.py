### Cochlear-Project Analysis file 
## Mingi Kang, Abhilasha Kumar 

import os, sys 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt




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
    'average',
    'only_w2v',
    'only_s2v'
]

models = [
        
    'forage_static',
    'forage_dynamic_simdrop',
    'forage_dynamic_multimodal_alpha=0.0',
    'forage_dynamic_multimodal_alpha=0.1',
    'forage_dynamic_multimodal_alpha=0.2',
    'forage_dynamic_multimodal_alpha=0.30000000000000004',
    'forage_dynamic_multimodal_alpha=0.4',
    'forage_dynamic_multimodal_alpha=0.5',
    'forage_dynamic_multimodal_alpha=0.6000000000000001',
    'forage_dynamic_multimodal_alpha=0.7000000000000001',
    'forage_dynamic_multimodal_alpha=0.8',
    'forage_dynamic_multimodal_alpha=0.9',
    'forage_dynamic_multimodal_alpha=1.0',
    'forage_dynamic_norms_associative',
    'forage_dynamic_norms_categorical',
    'forage_dynamic_delta_rise=0.0_fall=0.0',
    'forage_dynamic_delta_rise=0.0_fall=0.25',
    'forage_dynamic_delta_rise=0.0_fall=0.5',
    'forage_dynamic_delta_rise=0.0_fall=0.75',
    'forage_dynamic_delta_rise=0.0_fall=1.0',
    'forage_dynamic_delta_rise=0.25_fall=0.0',
    'forage_dynamic_delta_rise=0.25_fall=0.25',
    'forage_dynamic_delta_rise=0.25_fall=0.5',
    'forage_dynamic_delta_rise=0.25_fall=0.75',
    'forage_dynamic_delta_rise=0.25_fall=1.0',
    'forage_dynamic_delta_rise=0.5_fall=0.0',
    'forage_dynamic_delta_rise=0.5_fall=0.25',
    'forage_dynamic_delta_rise=0.5_fall=0.5',
    'forage_dynamic_delta_rise=0.5_fall=0.75',
    'forage_dynamic_delta_rise=0.5_fall=1.0',
    'forage_dynamic_delta_rise=0.75_fall=0.0',
    'forage_dynamic_delta_rise=0.75_fall=0.25',
    'forage_dynamic_delta_rise=0.75_fall=0.5',
    'forage_dynamic_delta_rise=0.75_fall=0.75',
    'forage_dynamic_delta_rise=0.75_fall=1.0',
    'forage_dynamic_delta_rise=1.0_fall=0.0',
    'forage_dynamic_delta_rise=1.0_fall=0.25',
    'forage_dynamic_delta_rise=1.0_fall=0.5',
    'forage_dynamic_delta_rise=1.0_fall=0.75',
    'forage_dynamic_delta_rise=1.0_fall=1.0',
    'forage_phonologicalstatic',
    'forage_phonologicaldynamicglobal_simdrop',
    'forage_phonologicaldynamiclocal_simdrop',
    'forage_phonologicaldynamicswitch_simdrop',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.0',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.0',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.0',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.1',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.1',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.1',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.2',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.2',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.2',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.30000000000000004',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.30000000000000004',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.30000000000000004',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.4',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.4',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.4',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.5',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.5',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.5',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.6000000000000001',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.6000000000000001',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.6000000000000001',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.7000000000000001',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.7000000000000001',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.7000000000000001',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.8',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.8',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.8',
    'forage_phonologicaldynamicglobal_multimodal_alpha=0.9',
    'forage_phonologicaldynamiclocal_multimodal_alpha=0.9',
    'forage_phonologicaldynamicswitch_multimodal_alpha=0.9',
    'forage_phonologicaldynamicglobal_multimodal_alpha=1.0',
    'forage_phonologicaldynamiclocal_multimodal_alpha=1.0',
    'forage_phonologicaldynamicswitch_multimodal_alpha=1.0',
    'forage_phonologicaldynamicglobal_norms_associative',
    'forage_phonologicaldynamiclocal_norms_associative',
    'forage_phonologicaldynamicswitch_norms_associative',
    'forage_phonologicaldynamicglobal_norms_categorical',
    'forage_phonologicaldynamiclocal_norms_categorical',
    'forage_phonologicaldynamicswitch_norms_categorical',
    'forage_phonologicaldynamicglobal_delta_rise=0.0_fall=0.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.0_fall=0.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.0_fall=0.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.0_fall=0.25',
    'forage_phonologicaldynamiclocal_delta_rise=0.0_fall=0.25',
    'forage_phonologicaldynamicswitch_delta_rise=0.0_fall=0.25',
    'forage_phonologicaldynamicglobal_delta_rise=0.0_fall=0.5',
    'forage_phonologicaldynamiclocal_delta_rise=0.0_fall=0.5',
    'forage_phonologicaldynamicswitch_delta_rise=0.0_fall=0.5',
    'forage_phonologicaldynamicglobal_delta_rise=0.0_fall=0.75',
    'forage_phonologicaldynamiclocal_delta_rise=0.0_fall=0.75',
    'forage_phonologicaldynamicswitch_delta_rise=0.0_fall=0.75',
    'forage_phonologicaldynamicglobal_delta_rise=0.0_fall=1.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.0_fall=1.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.0_fall=1.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.25_fall=0.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.25_fall=0.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.25_fall=0.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.25_fall=0.25',
    'forage_phonologicaldynamiclocal_delta_rise=0.25_fall=0.25',
    'forage_phonologicaldynamicswitch_delta_rise=0.25_fall=0.25',
    'forage_phonologicaldynamicglobal_delta_rise=0.25_fall=0.5',
    'forage_phonologicaldynamiclocal_delta_rise=0.25_fall=0.5',
    'forage_phonologicaldynamicswitch_delta_rise=0.25_fall=0.5',
    'forage_phonologicaldynamicglobal_delta_rise=0.25_fall=0.75',
    'forage_phonologicaldynamiclocal_delta_rise=0.25_fall=0.75',
    'forage_phonologicaldynamicswitch_delta_rise=0.25_fall=0.75',
    'forage_phonologicaldynamicglobal_delta_rise=0.25_fall=1.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.25_fall=1.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.25_fall=1.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.5_fall=0.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.5_fall=0.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.5_fall=0.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.5_fall=0.25',
    'forage_phonologicaldynamiclocal_delta_rise=0.5_fall=0.25',
    'forage_phonologicaldynamicswitch_delta_rise=0.5_fall=0.25',
    'forage_phonologicaldynamicglobal_delta_rise=0.5_fall=0.5',
    'forage_phonologicaldynamiclocal_delta_rise=0.5_fall=0.5',
    'forage_phonologicaldynamicswitch_delta_rise=0.5_fall=0.5',
    'forage_phonologicaldynamicglobal_delta_rise=0.5_fall=0.75',
    'forage_phonologicaldynamiclocal_delta_rise=0.5_fall=0.75',
    'forage_phonologicaldynamicswitch_delta_rise=0.5_fall=0.75',
    'forage_phonologicaldynamicglobal_delta_rise=0.5_fall=1.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.5_fall=1.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.5_fall=1.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.75_fall=0.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.75_fall=0.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.75_fall=0.0',
    'forage_phonologicaldynamicglobal_delta_rise=0.75_fall=0.25',
    'forage_phonologicaldynamiclocal_delta_rise=0.75_fall=0.25',
    'forage_phonologicaldynamicswitch_delta_rise=0.75_fall=0.25',
    'forage_phonologicaldynamicglobal_delta_rise=0.75_fall=0.5',
    'forage_phonologicaldynamiclocal_delta_rise=0.75_fall=0.5',
    'forage_phonologicaldynamicswitch_delta_rise=0.75_fall=0.5',
    'forage_phonologicaldynamicglobal_delta_rise=0.75_fall=0.75',
    'forage_phonologicaldynamiclocal_delta_rise=0.75_fall=0.75',
    'forage_phonologicaldynamicswitch_delta_rise=0.75_fall=0.75',
    'forage_phonologicaldynamicglobal_delta_rise=0.75_fall=1.0',
    'forage_phonologicaldynamiclocal_delta_rise=0.75_fall=1.0',
    'forage_phonologicaldynamicswitch_delta_rise=0.75_fall=1.0',
    'forage_phonologicaldynamicglobal_delta_rise=1.0_fall=0.0',
    'forage_phonologicaldynamiclocal_delta_rise=1.0_fall=0.0',
    'forage_phonologicaldynamicswitch_delta_rise=1.0_fall=0.0',
    'forage_phonologicaldynamicglobal_delta_rise=1.0_fall=0.25',
    'forage_phonologicaldynamiclocal_delta_rise=1.0_fall=0.25',
    'forage_phonologicaldynamicswitch_delta_rise=1.0_fall=0.25',
    'forage_phonologicaldynamicglobal_delta_rise=1.0_fall=0.5',
    'forage_phonologicaldynamiclocal_delta_rise=1.0_fall=0.5',
    'forage_phonologicaldynamicswitch_delta_rise=1.0_fall=0.5',
    'forage_phonologicaldynamicglobal_delta_rise=1.0_fall=0.75',
    'forage_phonologicaldynamiclocal_delta_rise=1.0_fall=0.75',
    'forage_phonologicaldynamicswitch_delta_rise=1.0_fall=0.75',
    'forage_phonologicaldynamicglobal_delta_rise=1.0_fall=1.0',
    'forage_phonologicaldynamiclocal_delta_rise=1.0_fall=1.0',
    'forage_phonologicaldynamicswitch_delta_rise=1.0_fall=1.0',
    'forage_random_baseline'
]


def mean_responses(): 
    data = pd.read_csv('../forager/data/fluency_lists/participant_results/participant_word_count.csv')
    
    participant_groupings = pd.read_csv("../forager/data/fluency_lists/participant_results/participant_groupings.csv")
    group_dict = dict(zip(participant_groupings['ID'], participant_groupings['Group']))
    
    # add group column to participant_word_count.csv
    participants = data['ID'].tolist()
    groupings = []
    for participant in participants: 
        groupings += [group_dict[participant]]
    data['Group'] = groupings
    
    data = data[data['Group'] != 'n/a']
    data.to_csv('../forager/data/fluency_lists/participant_results/participant_word_count.csv', index = False)

    # create graph of mean number of responses for each group (NH, CI)
    NH = data[data['Group'] == 'NH']
    CI = data[data['Group'] == 'CI']
    NH_mean_responses = NH['count'].mean()
    CI_mean_responses = CI['count'].mean()
    color = ['blue', 'red']
    
    print(NH_mean_responses)
    print(CI_mean_responses)
    # create graph 
    responses = [NH_mean_responses, CI_mean_responses]
    groups = ['NH', 'CI']
    
    plt.bar(groups, responses, color=color) 
    plt.xlabel('Participant Groups') 
    plt.ylabel('Mean Number of Responses')
    plt.title('Mean Number of Responses for Each Group')
    plt.savefig('outputs/mean_responses.png')
    

def add_groupings(): 
    participant_demographic = pd.read_csv('../forager/data/fluency_lists/participant_data/latest_demo.csv')

    ID = participant_demographic['ID'].tolist()
    Group = participant_demographic['Group'].tolist()
    group_dict = dict(zip(ID, Group))

    print(group_dict)
    
    # add group column to model_results.csv and individual_descriptive_stats.csv
    for dim in dimensions: 
        for t in type:         
            
            # model results 
            model_results_path = '../forager/output/' + dim + '_dim_results/' + t + '_results' +'/model_results.csv'
            model_results = pd.read_csv(model_results_path)
            
            ## get the list of subjects and create a new list for their groupings 
            forager_subjects = model_results["Subject"].tolist()
            forager_groupings = [] 
            
            for subject in forager_subjects: 
                forager_groupings += [group_dict[int(subject[4:])]]

            model_results['Group'] = forager_groupings

            model_results.to_csv(model_results_path, index = False)
            
            # individual descriptive stats.csv 
            individual_descriptive_stats_path = '../forager/output/' + dim + '_dim_results/' + t + '_results' +'/individual_descriptive_stats.csv'
            individual_descriptive_stats = pd.read_csv(individual_descriptive_stats_path)
            
            ## get the list of subjects and create a new list for their groupings 
            forager_subjects = individual_descriptive_stats["Subject"].tolist()
            forager_groupings = [] 

            for subject in forager_subjects: 
                forager_groupings += [group_dict[int(subject[4:])]]

            individual_descriptive_stats['Group'] = forager_groupings

            individual_descriptive_stats.to_csv(individual_descriptive_stats_path, index = False)

def statistics(): 
    NH = pd.DataFrame()
    CI = pd.DataFrame()
    
    for dim in dimensions:
        for t in type:
            path = '../forager/output/' + dim + '_dim_results/' + t + '_results' + '/individual_descriptive_stats.csv'
            data = pd.read_csv(path)
            data_groups = data.groupby(["Group"])
            
            NH_data = data_groups.get_group("NH")
            CI_data = data_groups.get_group("CI")
            
            NH = pd.concat([NH, NH_data])
            CI = pd.concat([CI, CI_data])
    
    NH.reset_index(drop=True, inplace=True)
    CI.reset_index(drop=True, inplace=True)

    # Calculate the means
    NH_Semantic_Similarity_mean = NH['Semantic_Similarity_mean'].mean()
    NH_Phonological_Similarity_mean = NH['Phonological_Similarity_mean'].mean()
    NH_Frequency_mean = NH['Frequency_Value_mean'].mean()
    
    
    CI_Semantic_Similarity_mean = CI['Semantic_Similarity_mean'].mean()
    CI_Phonological_Similarity_mean = CI['Phonological_Similarity_mean'].mean()
    CI_Frequency_mean = CI['Frequency_Value_mean'].mean()
    
    # Create for mean semantic similarity
    plt.figure()
    plt.bar(['NH', 'CI'], [NH_Semantic_Similarity_mean, CI_Semantic_Similarity_mean], color=['blue', 'red'])
    plt.xlabel("Groups")
    plt.ylabel("Mean")
    plt.title("Mean Semantic Similarity for NH and CI")
    plt.savefig('outputs/mean_semantic_similarity.png')

    # Create the second bar plot
    plt.figure()
    plt.bar(['NH', 'CI'], [NH_Phonological_Similarity_mean, CI_Phonological_Similarity_mean], color=['blue', 'red'])
    plt.xlabel("Groups")
    plt.ylabel("Mean")
    plt.title("Mean Phonological Similarity for NH and CI")
    plt.savefig('outputs/mean_phonological_similarity.png')

    # Create the third bar plot
    plt.figure()
    plt.bar(['NH', 'CI'], [NH_Frequency_mean, CI_Frequency_mean], color=['blue', 'red'])
    plt.xlabel("Groups")
    plt.ylabel("Mean")
    plt.title("Mean Frequency for NH and CI")
    plt.savefig('outputs/mean_frequency.png')
    
    return None


# def mean_semantic_similarity(): 
    
#     NH = pd.DataFrame()
#     CI = pd.DataFrame()
    
#     for dim in dimensions: 
#         for t in type: 
#             path = '../forager/output/' + dim + '_dim_results/' + t + '_results' +'/individual_descriptive_stats.csv'
#             data = pd.read_csv(path)
#             data_groups = data.groupby(["Group"])
            
#             NH_data = data_groups.get_group("NH")
#             CI_data = data_groups.get_group("CI")
            
#             NH = pd.concat([NH, NH_data])
#             CI = pd.concat([CI, CI_data])
    
#     NH.reset_index(drop=True, inplace=True)
#     CI.reset_index(drop=True, inplace=True)

#     NH_Semantic_Similarity_mean = NH['Semantic_Similarity_mean'].mean()
    
#     CI_Semantic_Similarity_mean = CI['Semantic_Similarity_mean'].mean()
    
#     # Creating the bar plot
#     plt.bar(['NH', 'CI'], [NH_Semantic_Similarity_mean, CI_Semantic_Similarity_mean], color=['blue', 'red'])

#     # Adding labels and title
#     plt.xlabel("Groups")
#     plt.ylabel("Mean")
#     plt.title("Mean Semantic Similarity for NH and CI")
#     plt.savefig('outputs/mean_semantic_similarity.png')
    
#     return None        

# def mean_phonological_similarity():
#     NH = pd.DataFrame()
#     CI = pd.DataFrame()
    
#     for dim in dimensions:
#         for t in type:
#             path = '../forager/output/' + dim + '_dim_results/' + t + '_results' + '/individual_descriptive_stats.csv'
#             data = pd.read_csv(path)
#             data_groups = data.groupby(["Group"])
            
#             NH_data = data_groups.get_group("NH")
#             CI_data = data_groups.get_group("CI")
            
#             NH = pd.concat([NH, NH_data])
#             CI = pd.concat([CI, CI_data])
    
#     NH.reset_index(drop=True, inplace=True)
#     CI.reset_index(drop=True, inplace=True)

#     NH_Phonological_Similarity_mean = NH['Phonological_Similarity_mean'].mean()
    
#     CI_Phonological_Similarity_mean = CI['Phonological_Similarity_mean'].mean()
    
#     # Creating the bar plot
#     plt.bar(['NH', 'CI'], [NH_Phonological_Similarity_mean, CI_Phonological_Similarity_mean], color=['blue', 'red'])
    
#     # Adding labels and title
#     plt.xlabel("Groups")
#     plt.ylabel("Mean")
#     plt.title("Mean Phonological Similarity for NH and CI")
#     plt.savefig('outputs/mean_phonological_similarity.png')
    
    
#     return None


# def mean_frequency():
#     NH = pd.DataFrame()
#     CI = pd.DataFrame()
    
#     for dim in dimensions:
#         for t in type:
#             path = '../forager/output/' + dim + '_dim_results/' + t + '_results' + '/individual_descriptive_stats.csv'
#             data = pd.read_csv(path)
#             data_groups = data.groupby(["Group"])
            
#             NH_data = data_groups.get_group("NH")
#             CI_data = data_groups.get_group("CI")
            
#             NH = pd.concat([NH, NH_data])
#             CI = pd.concat([CI, CI_data])
    
#     NH.reset_index(drop=True, inplace=True)
#     CI.reset_index(drop=True, inplace=True)

#     NH_Frequency_mean = NH['Frequency_Value_mean'].mean()
    
#     CI_Frequency_mean = CI['Frequency_Value_mean'].mean()

#     # Creating the bar plot
#     plt.bar(['NH', 'CI'], [NH_Frequency_mean, CI_Frequency_mean], color=['blue', 'red'])

#     # Adding labels and title  
#     plt.xlabel("Groups")
#     plt.ylabel("Mean")
#     plt.title("Mean Frequency for NH and CI")
#     plt.savefig('outputs/mean_frequency.png')
    
    
#     return None


def model_results(): 
    dimension_model_results = pd.DataFrame(columns=["dimension", "type", "model", "group", "sum_NLL", "mean_beta_semantic", "mean_beta_freq", "mean_beta_phon"])
    groups = ["NH", "CI"]
    stats = ["sum_NLL", "mean_beta_semantic", "mean_beta_freq", "mean_beta_phon"]
    stats_label = ["Negative_Log_Likelihood_Optimized", "Beta_Semantic", "Beta_Frequency", "Beta_Phonological"]

    rows= [] 

    for dim in dimensions: 
        for t in type: 
            path = '../forager/output/' + dim + '_dim_results/' + t + '_results' +'/model_results.csv'

            data = pd.read_csv(path)
            data_groups = data.groupby(["Group", "Model"])

            for group in groups: 
                for model in models: 
                    row = [dim, t, model, group]
                    test_data = data_groups.get_group((group, model))

                    j = 0 
                    while j < len(stats): 
                        if j == 0:
                            row += [test_data[stats_label[j]].sum()]
                        else: 
                            row += [test_data[stats_label[j]].mean()]
                        j += 1 
                    rows += [row]


    for line in rows: 
        dimension_model_results.loc[len(dimension_model_results)] = line 
    dimension_model_results.to_csv("outputs/model_results.csv", index = False)
    return None 
        
def best_NLL(): 
    groups = ["NH", "CI"]

    dimensional_model_results = pd.read_csv("outputs/model_results.csv")

    df = dimensional_model_results.groupby(["dimension", "model", "group"])

    dim_nll = pd.DataFrame(columns=["dimension","type", "model", "group", "sum_NLL", "mean_beta_semantic", "mean_beta_freq", "mean_beta_phon"])

    for group in groups: 
        for model in models: 
            test_data = df.get_group((50, model, group))
            dim_nll = pd.concat([dim_nll, test_data.nsmallest(1, 'sum_NLL')])

    dim_nll.reset_index(drop=True, inplace=True)
    dim_nll.to_csv("outputs/best_NLL.csv", index = False)
    return None 

def run_analysis(): 
    
    # create graph of mean number of responses for each group (NH, CI)
    mean_responses()
    
    # add group columns for model_results.csv and individual_descriptive_stats.csv
    add_groupings()
    
    # create bar graph for participant statistics 
    statistics()
    # mean_semantic_similarity()
    
    # mean_phonological_similarity()
    
    # mean_frequency()
    
    # create model_results.csv for sum_NLL, mean_beta_semantic, mean_beta_freq, mean_beta_phon
    model_results()
    
    # create best_NLL.csv for best sum_NLL for each model and switch method (ex. dynamic_simdrop with alpha 0.5 s2v has lowest sum_NLL)
    best_NLL()
    return None 


run_analysis()