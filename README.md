# hatespeech
NLP course project

## Instructions for our implementation
### Task 1-3 (Wordcloud, empath categorization and Harward general inquirer)
1. Install dependencies (most should be in requirements.txt)
2. Run the jupyter notebook `jupyter notebook hatespeech.ipynb`
3. Open the notebook in your browser and run the cells in order

### Task 4-5 (Tensorflow neural network as a heuristic) 
4. Continue running the notebook from tasks 1-3

### Task 6
5. to be done

### Task 7
6. Clone the repository: https://github.com/younggns/comparative-abusive-lang
7. Change line 24 from data_preprocess.py from `return re.sub(pattern, repl, text, flags=FLAGS)` to `return re.sub(str(pattern), repl, str(text), flags=FLAGS)`
8. run data_preprocess.py
9. run train_ml_models.py --clf NB
10. input values are 1,3 and 1200 when they are asked

## Tasks
1. Consult the database of hate speech database maintained at http://hatespeechdata.com/ . Consider the dataset CONAN in the database, which has 1% abusive content among 1,288 posts. We would like to investigate the structure of the dataset in terms of categories present. Consider the subclass S1 of abusive content and subclass S2 of non-abusive content. Draw a wordcloud representation of S1 and S2.
2. Use Empath categorization and output for each message in S1 and S2, the list of categories who have non-zero weight. Report the result in a database that you will provide as a project deliverable. Conclude whether Empath categorization can be considered as a discriminative feature to recognize hate content from non-hate content.
3. Repeat 2) by using Harvard General Inquirer available in http://www.wjh.harvard.edu/~inquirer/inquirerbasic.xls
4. Suggest a simple heuristic that uses Empath categorization and General Inquirer that would allow you to identify the presence of hate content.
5. Evaluate the performance of this heuristic using ground truth of CONAN dataset.  You can also test this heuristic in another dataset of your choice of hatespeechdata.com 
6. Study the implementation available at https://github.com/pinkeshbadjatiya/twitter-hatespeech of the paper “Deep learning for hate speech detection Tweets” by Pinkesh Badjatiya (www’17 proceedings, 2017) and demonstrate that you closely reproduce the performance claimed by the authors in their paper. If not, comment on your findings accordingly. 
7. Study the implementation highlighted in https://github.com/younggns/comparative-abusive-lang but when using dataset of 5) or 4), report the corresponding accuracy, F1 and precision/recall.
8. Study the paper “Improving Hate Speech Detection with Deep Learning Ensembles” by Zimermann et al. LREC 2018), which uses the same dataset (Racisim, Sexism and none) and its implementation available at https://github.com/stevenzim/lrec-2018. Demonstrate that the program is working appropriately, and discuss its performances with respect to the two other previous implementations. 
9. We would like to test the above four implementations (three above papers and heuristic) using different datasets. For this purpose select another dataset from other hatespeechdata.com repositories and check that your program can run successfully.
10. Suggest a GUI of your own that allows us to exemplify the different steps above.
