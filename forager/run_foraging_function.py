def run_foraging_function(dimension, type):

    import argparse
    from scipy.optimize import fmin
    from forager.foraging import forage
    from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_norms_associative, switch_norms_categorical
    from forager.cues import create_history_variables
    from forager.utils import prepareData
    import pandas as pd
    import numpy as np
    from scipy.optimize import fmin, minimize
    import os, sys
    from tqdm import tqdm
    import warnings 
    import zipfile
    
    normspath =  'forager/data/norms/animals_snafu_scheme_vocab.csv'
    similaritypath =  'forager/data/lexical_data/' + dimension + '_dim_lexical_data/' + type + '/semantic_matrix.csv'
    frequencypath =  'forager/data/lexical_data/' + dimension + '_dim_lexical_data/' + type + '/frequencies.csv'
    phonpath = 'forager/data/lexical_data/' + dimension + '_dim_lexical_data/' + type + '/phon_matrix.csv'
    vocabpath = 'forager/data/lexical_data/Extras/vocab.csv'
    

    #  – pipeline models –switch simdrop –model all 
    
    
    model = 'dynamic'
    switch_method = 'simdrop'
    data = '/Users/mkang2/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/cochlear-project/forager/new_data.txt'
    
    def retrieve_data(path):
        """
        1. Verify that data path exists

        """
        if os.path.exists(path) == False:
            ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path".format(path=path)
            raise Exception(ex_str)
        data = prepareData(path)
        return data

    def get_lexical_data():
        norms = pd.read_csv(normspath, encoding="unicode-escape")
        similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
        frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
        phon_matrix = np.loadtxt(phonpath,delimiter=',')
        labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
        return norms, similarity_matrix, phon_matrix, frequency_list,labels
    
    def calculate_model(history_vars, switch_names, switch_vecs):
        """
        1. Check if specified model is valid
        2. Return a set of model functions to pass
        """
        model_name = []
        model_results = []
        
        for i, switch_vec in enumerate(switch_vecs):
            r1 = np.random.rand()
            r2 = np.random.rand()

            v = minimize(forage.model_dynamic, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec)).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            
            nll, nll_vec = forage.model_dynamic_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1],switch_vec)
            model_name.append('forage_dynamic_' + 'simdrop')
            model_results.append((beta_df, beta_ds, nll, nll_vec))

        # Unoptimized Model
        model_name.append('forage_random_baseline')
        nll_baseline, nll_baseline_vec = forage.model_static_report(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1])
        model_results.append((0, 0, nll_baseline, nll_baseline_vec))
        return model_name, model_results
    
    
    def calculate_switch(fluency_list, semantic_similarity):
        '''
        1. Check if specified switch model is valid
        2. Return set of switches, including parameter value, if required

        switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','all']
        '''
        switch_names = []
        switch_vecs = []

        switch_names.append(switch_method)
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))    
        
        return switch_names, switch_vecs


    def run_model(data, model_type, switch_type):
        # Get Lexical Data needed for executing methods
        norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
        forager_results = []
        # Run through each fluency list in dataset
        for i, (subj, fl_list) in enumerate(tqdm(data)):
            print("\nRunning Model for Subject {subj}".format(subj=subj))
            import time
            start_time = time.time()
            # Get History Variables 
            history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
            
            # Calculate Switch Vector(s)
            switch_names, switch_vecs = calculate_switch(fl_list, history_vars[0])

            #Execute Individual Model(s) and get result(s)
            model_names, model_results = calculate_model(history_vars, switch_names, switch_vecs)

            #Create Model Output Results DataFrame
            for i, model in enumerate(model_names):
                model_dict = dict()
                model_dict['Subject'] = subj
                model_dict['Model'] = model
                model_dict['Beta_Frequency'] = model_results[i][0]
                model_dict['Beta_Semantic'] = model_results[i][1]
                # print(results[i])
                # sys.exit()
                if len(model_results[i]) == 4:
                    model_dict['Beta_Phonological'] = None
                    model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][2]
                if len(model_results[i]) == 5:
                    model_dict['Beta_Phonological'] = model_results[i][2]
                    model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][3]
                forager_results.append(model_dict)
        forager_results = pd.DataFrame(forager_results)
            
        return forager_results
    
    def run_lexical(data):
        # Get Lexical Data needed for executing methods
        norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
        lexical_results = []
        for i, (subj, fl_list) in enumerate(tqdm(data)):
            history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
            lexical_df = pd.DataFrame()
            lexical_df['Subject'] = len(fl_list) * [subj]
            lexical_df['Fluency_Item'] = fl_list
            lexical_df['Semantic_Similarity'] = history_vars[0]
            lexical_df['Frequency_Value'] = history_vars[2]
            lexical_df['Phonological_Similarity'] = history_vars[4]
            lexical_results.append(lexical_df)
        lexical_results = pd.concat(lexical_results,ignore_index=True)
        return lexical_results
    
    def run_switches(data):
        norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
        switch_results = []
        for i, (subj, fl_list) in enumerate(tqdm(data)):
            history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
            switch_names, switch_vecs = calculate_switch(fl_list, history_vars[0])
        
            switch_df = []
            for j, switch in enumerate(switch_vecs):
                df = pd.DataFrame()
                df['Subject'] = len(switch) * [subj]
                df['Fluency_Item'] = fl_list
                df['Switch_Value'] = switch
                df['Switch_Method'] = switch_names[j]
                switch_df.append(df)
        
            switch_df = pd.concat(switch_df, ignore_index=True)
            switch_results.append(switch_df)
        switch_results = pd.concat(switch_results, ignore_index=True)
        return switch_results
    
    
    def indiv_desc_stats(lexical_results, switch_results = None):
        metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
        metrics.replace(.0001, np.nan, inplace=True)
        grouped = metrics.groupby('Subject').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)
        num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
        grouped['#_of_Items'] = num_items[grouped['Subject']].values
        # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
        if switch_results is not None:
            # count the number of unique values in the Switch_Method column of the switch_results DataFrame
            n_rows = len(switch_results['Switch_Method'].unique())
            new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

            # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times

            new_df.iloc[(slice(None, None, n_rows)), :] = grouped
            new_df['Subject'] = new_df['Subject'].ffill()

            switch_methods = []
            num_switches_arr = []
            cluster_size_mean = []
            cluster_size_sd = []
            for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
                switch_method = sub[1]
                cluster_lengths = []
                num_switches = 0
                ct = 0
                for x in fl_list['Switch_Value'].values:
                    ct += 1
                    if x == 1:
                        num_switches += 1
                        cluster_lengths.append(ct)
                        ct = 0
                if ct != 0:
                    cluster_lengths.append(ct)
                avg = sum(cluster_lengths) / len(cluster_lengths)
                sd = np.std(cluster_lengths)
                switch_methods.append(switch_method)
                num_switches_arr.append(num_switches)
                cluster_size_mean.append(avg)
                cluster_size_sd.append(sd)

            new_df['Switch_Method'] = switch_methods
            new_df['Number_of_Switches'] = num_switches_arr
            new_df['Cluster_Size_mean'] = cluster_size_mean
            new_df['Cluster_Size_std'] = cluster_size_sd
            grouped = new_df
            
        return grouped
    
    def agg_desc_stats(switch_results, model_results=None):
        agg_df = pd.DataFrame()
        # get number of switches per subject for each switch method
        switches_per_method = {}
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            method = sub[1]
            if method not in switches_per_method:
                switches_per_method[method] = []
            if 1 in fl_list['Switch_Value'].values:
                switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
            else: 
                switches_per_method[method].append(0)
        agg_df['Switch_Method'] = switches_per_method.keys()
        agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
        agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
        
        if model_results is not None:
            betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
            betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
            grouped = betas.groupby('Model').agg(['mean', 'std'])
            grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
            grouped.reset_index(inplace=True)

            # add a column to the grouped dataframe that contains the switch method used for each model
            grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
            # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
            switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
            for model in switch_models:
                # replace only the second underscore with a space
                grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
                grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
            
            # split the Model column on the space
            grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

            # merge the two dataframes on the Switch_Method column 
            agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')

        return agg_df
        
    oname = 'forager/output/' + dimension + '_dim_results/' + type + '_results.zip'

    
    
    switch_name = 'switch_results.csv'
    lexical_name = 'lexical_results.csv'
    models_name = 'model_results.csv'
    
    # Check for model and switch parameters
    print("Checking Data ...")
    data, replacement_df, processed_df = retrieve_data(data)
    
    print("Retrieving Lexical Data ...")
    lexical_results = run_lexical(data)
    
    print("Obtaining Switch Designations ...")
    switch_results = run_switches(data)
    print("Running Forager Models...")
    forager_results = run_model(data, model, switch_method)

    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    agg_stats = agg_desc_stats(switch_results, forager_results)
    with zipfile.ZipFile(oname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)
        
        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)
        # save lexical results
        with zipf.open(lexical_name,'w') as csvf:
            lexical_results.to_csv(csvf, index=False) 
        # save switch results
        with zipf.open(switch_name,'w') as csvf:
            switch_results.to_csv(csvf, index=False) 
        # save model results
        with zipf.open(models_name,'w') as csvf:
            forager_results.to_csv(csvf, index=False) 
        # save individual descriptive statistics
        with zipf.open('individual_descriptive_stats.csv', 'w') as csvf:
            ind_stats.to_csv(csvf, index=False)
        # save aggregate descriptive statistics
        with zipf.open('aggregate_descriptive_stats.csv', 'w') as csvf:
            agg_stats.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")
        print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{oname}'")
        print(f"File 'model_results.csv' containing model level NLL results of provided fluency data saved in '{oname}'")
        print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
        print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{oname}'")



dimensions = ['50', '100', '200', '300']
type = [
    'only_s2v', 
    'only_w2v'
    # 'alpha_0_s2v', 
    # 'alpha_0_w2v', 
    # 'alpha_0.1_s2v', 
    # 'alpha_0.1_w2v', 
    # 'alpha_0.2_s2v', 
    # 'alpha_0.2_w2v', 
    # 'alpha_0.3_s2v', 
    # 'alpha_0.3_w2v', 
    # 'alpha_0.4_s2v', 
    # 'alpha_0.4_w2v', 
    # 'alpha_0.5_s2v', 
    # 'alpha_0.5_w2v', 
    # 'alpha_0.6_s2v', 
    # 'alpha_0.6_w2v',
    # 'alpha_0.7_s2v', 
    # 'alpha_0.7_w2v', 
    # 'alpha_0.8_s2v', 
    # 'alpha_0.8_w2v', 
    # 'alpha_0.9_s2v', 
    # 'alpha_0.9_w2v', 
    # 'alpha_1.0_s2v', 
    # 'alpha_1.0_w2v',
    # 'average'
]

#test
# run_foraging_function('50', 'only_s2v')

count = 0 
for dim in dimensions: 
    for t in type: 
        run_foraging_function(dim, t)
        count += 1

print(count)