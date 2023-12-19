# cochlear-project
'cochlear-project' is a Lexicon Lab's project trying to answer the similarities and differences in memory search and memory organization for Normal Hearing and Cochlear Implant individuals. This project utilizes the 'forager' package that analyzes verbal fluency data and implementing models of memory search with the use of two language models: Speech2Vec and Word2Vec. The run through of the project is in the description below. 


## Project Details 

### Project Organization 
- The 'analysis' folder consists of all the analysis this project achieved by using/modifying the forager package. In the analysis folder, there are two  files 'workflow.py' and 'analysis.py'. 
    - workflow.py - The start of the cochlear project was analyzing the original participants' SFT (semantic fluency task) responses for animals. There were many out of vocabulary words that did not have an embedding from Word2Vec/Speech2Vec. We were able to solve this problem with alternatives from animals_snafu_scheme.csv categorical replacements and spell checking for certain animal words from the participants' responses. Then we are able to get the lexical data that is used in forager to run the forager to get the results of the participant data. In the forager folder, the data folder contains all the lexical data of different dimensions and alpha values (combinations of Speech2Vec and Word2Vec) for the embeddings. 
        - cochlear-project/forager/data/lexical_data 
            - lexical data for different dimensions and alpha values (embeddings.csv, semantic_matrix.csv, phonological_matrix.csv, frequencies.csv)
        - cochlear-project/forager/data/fluency_lists/participant_data: 
            - raw-data.txt (original participant data)
            - transformed-data.csv/.txt (modified participant data for forager usage)
            - latest_demo.csv (participant demographics)
            - animal_snafu_scheme.csv (categories for animals)
        - cochlear-project/forager/data/fluency_lists/participant_results: 
            - data-cochlear-replacements.csv (table of what words were replaced with what)
            - participant_groupings.csv (participant designation of either NH, CI)
            - participant_word_count.csv (number of responses produced by each participant)
    - analysis.py - After getting the lexical data for the different dimensions and alpha values, we run the modified participant data through forager. This file answers the question of: 
        - How many itmes each group generated in response in CI/NH? - analysis/outputs/mean_responses.png
        - What is the mean semantic similarity, phonological similarity, frequency for the two groups (NH, CI), average for all possible embeddings. - analysis/outputs/mean_statistics.png
        - What is the best switch-method/model/alpha-value/dimensions for each group? - analysis/outputs/model_results.csv & analysis/outputs/best_NLL.csv
    - analysis/plots - This folder contains a couple visualization for earlier analysis for Speech2Vec vs. Word2Vec, NH vs. CI, and comparing different dimensions for the embeddings. 
    

### Running the analysis
- To run the experiment and get the analysis, go to workflow.py and run the file. This should clean the original participant data/make replacements for words, and run the analysis.py file from workflow.py to get the analysis files that are located in analysis/outputs.
- Notes: Creating the Phonological Matrix may take a couple hours, 
    



## Forager Functionality

### Outputs

Different output files are generated when forager is run, based on the use case:

- **Evaluate/Check Data**: If the user first evaluates the data they wish to analyze, the following files will be available for download via a `.zip` file:
  - `evaluation_results.csv`: This file will contain the results from *forager*'s evaluation of all the items in the data against its own vocabulary. The `evaluation` column in the file will describe whether an exact match to the item was found (FOUND), or a reasonable replacement was made (REPLACE), and how the OOV items were handled based on the user-specified policy (EXCLUDE/TRUNCATE/UNK). The `replacement` column in the file will describe the specific replacements made to the items.
  - `processed_data.csv`: This file will contain the final data that will be submitted for further analyses to *forager*. Users should carefully inspect this file to ensure that it aligns with their expectations, and make changes if needed.
  - `forager_vocab.csv`: This file contains the vocabulary used by *forager* to conduct the evaluation. We provide this file so that users can make changes to their data if needed before they proceed with other analyses.

- **Get Lexical Values**: If the user selects this option, the following files will be available for download via a `.zip` file:

  - `lexical_results.csv`: This file contains item-wise lexical metrics (semantic similarity, phonological similarity, and word frequency). The semantic and phonological similarities indicate the similarity between the previous item and current item (the first item will have an arbitrary value of .0001),  whereas the frequency values indicate the frequency of the current item in the English language (obtained via Google N-grams).
  - `individual_descriptive_stats.csv`: This file contains some aggregate metrics at the participant level such as total number of items produced, as well as means/SDs of semantic similarity, phonological similarity, and word frequency.
  - The three files from the evaluation phase

- **Get Switches**: If the user selects this option, the following files will be available for download via a `.zip` file:

  - `switch_results.csv`: This file contains the item-level cluster/switch designations for each method. A switch is indicated by a 1, and a cluster is indicated by a 0. A value of 2 either denotes the first item in the list or the last item(s) for switch methods that rely on previous/subsequent items (i.e., no switch/cluster prediction can be made).
  - `lexical_results.csv`: This file will be identical to the one generated in the **Get Lexical Values** option.
  - `individual_descriptive_stats.csv`: In addition to the metrics available from lexical results (mean/SD of lexical values and number of items), this file will also contain the total number of switches and mean cluster size for each switch method.
  - `aggregate_descriptive_stats.csv`: This file will contain mean/SD for the number of switches (aggregated across all participants) for each switch method.
  - The three files from the evaluation phase

- **Get Models**: If the user selects this option, they will be redirected to a Colab notebook, where they can upload their data and run the models. In addition to the files generated in the **Get Switches** option, the following files will be available for download via a `.zip` file:

  - `model_results.csv`: This file will contain the model-based negative log-likelihoods for the selected models, as well as the best-fitting parameter values for semantic, phonological, and frequency cues for each model, at the subject level.
  - `aggregate_descriptive_stats.csv`: In addition to the metrics available from the **Get Switches** option, this file will also contain the mean/SD values of the parameters (aggregated across all participants) for each model and switch method.


### Semantic Foraging Models

The source code for these models can be found inside `forager/foraging.py`. We currently implement four types of semantic foraging models, which can be executed by passing the corresponding model name to the ```--model``` flag in the command line interface. The models are as follows:
- Static Model 
    - the original static model (```static```) executes foraging where all transitions are based on the same set of cues over the entire retrieval interval, effectively ignoring the patchy structure of the memory retrieval environment. All transitions are based on a combined product of semantic similarity and word frequency. The static foraging model was introduced in Hills TT, Jones MN, Todd PM (2012).
- Dynamic Model
    - the original dynamic model (```dynamic```) executes foraging by employing a clustering and switching mechanism that exploits the patchy structure of memory. The dynamic model utilizes word frequency and semantic similarity during within-clustering transitions, and word frequency during between-cluster transitions. Cluster and switching behavior is captured via the ```switchvals``` parameter, which can be calculated by the provided switch methods in the package. The static foraging model was introduced in Hills TT, Jones MN, Todd PM (2012).
- Phonological Static Model
    - the phonological static model (```pstatic```) is an extension of the static model, where all transitions are based on a combined product of semantic similarity, word frequency, and phonological similarity. The phonological static model was introduced in Kumar AA, Lundin NB, & Jones MN (2022)
- Phonological Dynamic Model
    - the phonological dynamic (```pdynamic```) model has 3 versions, indexed by the ```phoncue``` parameter. The "local" model uses frequency, semantic, and phonological similarity during within-cluster transitions and frequency during between-cluster transitions. The "global" model uses frequency, semantic, and phonological similarity during within-cluster transitions, and frequency and phonological similarity during between-cluster transitions. Finally, the "switch" model uses only semantic similarity and frequency during within-cluster transitions and phonological similarity and frequency for between-cluster transitions. By default, if using run_foraging.py, if ```pdynamic``` is passed to --model flag, it will execute all three versions of the model. The phonological dynamic model was introduced in Kumar AA, Lundin NB, & Jones MN (2022)

### Switch Methods
The source code for these methods can be found inside `forager/switch.py`. We currently implement four types of switch methods, which can be executed by passing the corresponding switch name to the ```--switch``` flag in the command line interface. The methods are as follows:
- Norms-based (Troyer Norms)
    - the troyer norms switching method (```troyer```) adapts the categorization norms proposed by Troyer, AK, Moscovitch, M, & Winocur, G (1997), subsequently extended by our lab for analysis. Switches are predicted when moving from one subcategory of the Troyer categorization norms to another.  
- Similarity Drop
    - the similarity drop switching method (```simdrop```) is based on a switch heuristic used in Hills TT, Jones MN, Todd PM (2012) to mimic optimal foraging behavior. A switch is predicted within a series of items A,B,C,D after B if S(A,B) > S(B,C) and S(B,C) < S(C,D).
- Delta Similarity
    - the delta similarity switching method (```delta```) is a switch method proposed by Nancy Lundin in her dissertation to bypass the limits of the similarity drop switching method by allowing for consecutive switches and accounting for small dips in similarity that similarity drop may deem as a switch. This is done through the inclusion of z-scoring semantic similarity across all transitions in a list, and the inclusion of rise and fall threshold parameters to control clustering and switching via thresholding on z-score similarity values.
- Multimodal Similarity Drop
    - the multimodal similarity drop switching method (```multimodal```) is a switch method developed to include phonological similarity into the switch heuristic proposed by Hills TT, Jones MN, Todd PM (2012). It includes an alpha parameter which dictates the weighting of semantic versus phonological similarity in switching from cluster to cluster.

### Cues (Semantic, Phonological, and Frequency Matrix) Generation

The source code for these methods can be found inside `forager/cues.py`. We currently implement three types of cue generation methods, which are as follows:

Semantic Similarity Matrix Generation
- The semantic similarity matrix is generated using an underlying semantic representational model ("embeddings"). The package currently uses the word2vec model and computes pairwise cosine similarity for all items in the space (indexed by the size of embeddings).

Phonological Matrix Generation
- The phonological similarity matrix computes the pairwise normalized edit distance between the phonemic transcriptions of all items in the space (indexed by a list (```labels```). Phonemic transcriptions are obtained via CMUdict, which uses Arpabet phonemic transcriptions.

Frequency Data Generation
- A table of item frequencies is generated by obtaining raw counts for each item in the embedding labels from the Google Books Ngram Dataset via the PhraseFinder API. The raw counts are log transformed, and these log counts are the metrics used later by the models.

History Variabile Creation:
- History variables is a utility function that keeps track of lexical metrics (frequency, semantic, and phonological similarity) within a given fluency list. Specifically, the function uses underlying semantic and phonological similarity matrices as well as word frequency, and returns the similarites between consecutive items within a specific fluency list.

### Lexical Metrics (Embeddings and Frequency)

We also provide functions to obtain embeddings and frequency data for a given vocabulary set. The source code for these methods can be found inside `forager/embeddings.py` and `forager/frequency.py`. 

Embeddings
- We use the `pymagnitude` packagae to obtain word vector embeddings. Currently, we use the word2vec model trained on the GoogleNews corpus that produces 300-dimensional word embeddings. `pymagnitude` also provides other embedding models.

Frequency
- We use the Google Books Ngram Dataset to obtain word frequency data. The package provides a function to obtain raw counts for a given vocabulary set. The raw counts are log transformed, and these log counts are the metrics used later by the models.

### Util Functions (Data Preprocessing)

The source code for this data preprocessing method can be found inside `forager/utils.py`. 
Prepare Data Function
- The data preparation function cleans and reformats the fluency list data provided by the user. It takes in a path to data in the form of a file in which the first column contains a participant ID and the second contains one response item. The first row is assumed to be a header. If the file has more than two columns, users will be given the option to use the third as the timepoint for the fluency list (i.e., if a participant has multiple lists). Accepted delimiters separating the columns include commas, tabs, semicolons, pipes, and spaces. Each row should be on its own line. The function checks for any items outside of the vocabulary set used in the lexical metrics (OOV items). If a reasonable replacement is found for an OOV item, the item will be automatically replaced with the closest match. To handle all other OOV words, the user will be given three options. First, they can truncate the fluency list at the first occurrence of such a word. Second, they can exclude any such words but continue with the rest of the list, as if that word was never produced. Third, the word can be assigned a mean semantic vector, mean phonological similarity, and 0.0001 frequency. A file outlining the edits made to the original data will be saved. The fluency data is then reformatted into a list of tuples, each containing the participant ID and the corresponding fluency list. 



## Development Notes

## References

Please cite the following work if you use the package:
- Kumar, A.A., Apsel, M., Zhang, L., Xing, N., Jones. M.N. (2023). forager: A Python package and web interface for modeling mental search.
- Hills, T. T, Jones, M. N, & Todd, P. M (2012). Optimal foraging in semantic memory. *Psychological Review*, *119*(2), 431–440.
- Kumar, A. A, Lundin, N. B, & Jones, M. N (2022). Mouse-mole-vole: The inconspicuous benefit of phonology during retrieval from semantic memory. *Proceedings of the Annual Meeting of the Cognitive Science Society*. 
- Troyer A. K, Moscovitch M., Winocur G. (1997). Clustering and switching as two components of verbal fluency: evidence from younger and older healthy adults. *Neuropsychology*. Jan;11(1):138-46.
