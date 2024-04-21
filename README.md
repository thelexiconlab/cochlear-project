# Structure and process interactions in memory search for normal hearing and cochlear implant individuals

## Description

This project investigates how individuals search through memory, and how different semantic representations and search processes combine to produce search outcomes. We used a foraging model to simulate the search process and compared the model's predictions to data from normal hearing and cochlear implant participants. Additionally, to instantiate the semantic representations, we used word2vec and speech2vec models to examine the relative contributions of semantic and phonological information to the search process.

This repository hosts the code and data for the following paper presented at the Annual Meeting of the Cognitive Science Society:


- Kumar, A.A., Kang, M., Kronenberger, W.G., Pisoni, D. Jones, M.N. (in press). Structure and process-level lexical interactions in memory search: A case study of individuals with cochlear implants and normal hearing.


## Organization

### forager folder

- The `forager` folder consists of the forager package that was modified to fit the needs of this project. The original forager package can be found [here](https://github.com/thelexiconlab/forager/tree/master). This package was modified to construct different lexical data based on the word2vec and speech2vec models. 
- Details about the word2vec model can be found [here](https://arxiv.org/abs/1301.3781)
- Details about the speech2vec model can be found [here](https://arxiv.org/abs/1803.08976)

### analysis folder
- The `analysis` folder consists of all the analysis this project achieved by using/modifying the forager package. 
- The `cochlear-analysis.Rmd` file uses outputs generated via the forager package to conduct the analysis. These outputs can be found in the `forager/output` folder and are separated by whether the outputs were generated based on the full fluency lists OR the truncated fluency lists (to control for list length, see paper for more details).
- The `analysis/plots` folder consists of all the plots generated from the analysis conducted within the .Rmd file.

### error check folder

- Contains some unit tests for the forager package and resulting outputs.

## Contact

For any questions or concerns, please contact [Abhilasha Kumar](https://thelexiconlab.github.io/people/).