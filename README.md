# Developing NLP-based Search and Predict Tools for Wikipedia Articles

The goal of this project was to use NLP to develop search and predict tools for keywords and new articles introduced to a collection of Wikipedia articles based on category. I developed four python script files:

1. download: to use the Wikipedia API to collect articles based on category (with or without subcategory) classifications from Wikipedia. 
2. search: to perform a search of the collected articles and return the 5 most relevant articles in the collection.
3. train-model: to a train a model on the downloaded articles to predict the category classification of a new article that is introduced into the collection. 
4. predict: to predict the category classification of a new article that is introduced into the collection. 

The functions in these scripts are in the wikipedia_helper.py file.

There are two jupyter notebook that show the workflow of the files:
1. wikipedia_nlp_no_subcategories
2. wikipedia_nlp_subcategories

The sklearn methods I used included TFIDF, SVD, NearestNeighbors (search) and KNeighbors (predict). In this project, I used MongoDB for data storage and Redis for model storage and for temporary storage of other data.

Next updates: Adding comments to the wikipedia_helper file and the jupyter notebooks.
