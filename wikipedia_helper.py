import re
import requests
import pandas as pd
import numpy as np
import pymongo
import wikipedia
from spacy.en import English, STOP_WORDS
from IPython.display import display
import copy
from functools import reduce
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier


def execute_json_query(category):
    category = re.sub('_','+',category)
    query = 'http://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category%3A+{}&cmlimit=max'.format(category)
    r = requests.get(query)
    return r.json()


def json_to_dataframe(json_dict):
    df = pd.DataFrame(json_dict['query']['categorymembers'])
    if 'ns' in df.columns:
        df = df.drop('ns',axis=1)
    return df


def get_subcategories(dataframe, category_list,append_subcats=False):
    subcategory_list = []
    for item in dataframe['title'][dataframe['title'].str.contains('Category:')]:
        _, title = item.split(':',maxsplit=1)
        subcategory_list.append(title)
        if append_subcats == True:
            if title not in category_list:
                category_list.append(title)
    return subcategory_list


def get_article_category_pageid_titles(parent_categories,collector,append_subcats=False):
    category_dataframes = []
    cat_subcat_dict = {}
    uncollected_articles = {}
    for parent in parent_categories:
        categories = [parent]
        title_dataframes = []
        for cat in categories:
            js_dict = execute_json_query(cat)
            id_title_df = json_to_dataframe(js_dict)
            try:
                if append_subcats == True:
                    id_title_df['subcategory'] = [cat]*id_title_df.shape[0]
                id_title_df['category'] = [parent]*id_title_df.shape[0]
                subcategories = get_subcategories(id_title_df,categories,append_subcats)
                cat_subcat_dict[cat] = subcategories
                id_title_df = id_title_df[~id_title_df['title'].str.contains('Category:')]
                id_title_df = id_title_df[~id_title_df['title'].str.contains('File:')]
                title_dataframes.append(id_title_df)
            except:
                pass
        if title_dataframes == []:
            pass
        else:
            id_title_df = reduce(lambda left,right: pd.concat([left,right],ignore_index=True), title_dataframes)
            id_title_df = id_title_df[~id_title_df['pageid'].duplicated() == True]
            category_dataframes.append(id_title_df)
            print('\nCollecting articles for {}'.format(parent))
            uncollected = articles_to_collector(id_title_df,collector)
            uncollected_articles[parent] = uncollected
    if category_dataframes == []:
        pass
    else:
        id_title_df = reduce(lambda left,right: pd.concat([left,right],ignore_index=True), category_dataframes)
    for parent in parent_categories:
        print('{} has {} articles.'.format(parent,id_title_df[id_title_df['category'] == parent].shape[0]))
    print('The total number of articles found is {}.'.format(id_title_df['category'].shape[0]))
    print('The total number of articles stored is {}.'.format(collector.count()))
    return id_title_df, cat_subcat_dict, uncollected_articles


def articles_to_collector(article_titles_df,collector,append_subcats=False):
    indices_not_collected = []
    pages_not_collected = []
    for i in tqdm(article_titles_df.index):
        try:
            article_dict = {'pageid' :str(article_titles_df['pageid'][i]),
                            'category': article_titles_df['category'][i],
                            'title' : article_titles_df['title'][i]}
            if append_subcats == True:
                article_dict['subcategory'] = article_titles_df['subcategory'][i]
            article_dict['clean_text'] = wikipedia.page(title=article_dict['title'],
                                                        pageid=int(article_dict['pageid'])).content
            collector.insert_one(article_dict)
        except:
            pages_not_collected.append(article_titles_df['pageid'][i])
            pass
    print('number of articles found:', article_titles_df.shape[0])
    print('number of articles stored:', collector.count(),'\n')
    return pages_not_collected

    
def load_wiki_docs(find_cursor,collector_count):
    stored_docs = []
    for i in tqdm(range(collector_count)):
        doc = find_cursor.next()
        stored_docs.append(doc)
    docs_df = pd.DataFrame(stored_docs)    
    return docs_df


def get_clean_mongo_dataframe(mongo_dataframe):
    mongo_dataframe = mongo_dataframe[~mongo_dataframe['pageid'].duplicated() == True]
    return mongo_dataframe.drop(['_id'],axis=1)


def fit_svd_for_lsa(doc_texts_array,mindf=1,n=100,rand_state=12):
    tfidf_vectorizer = TfidfVectorizer(min_df = mindf, stop_words = 'english')
    doc_term_matrix = tfidf_vectorizer.fit_transform(doc_texts_array)
    SVD = TruncatedSVD(n_components=n,random_state=rand_state)
    lsa_matrix = SVD.fit_transform(doc_term_matrix)
    return tfidf_vectorizer, SVD, lsa_matrix


def preprocess_search_text(search_text,tfidf_vectorizer,svd_fit):
    search_text_vec = tfidf_vectorizer.transform([search_text])
    search_text_lsa = svd_fit.transform(search_text_vec)
    return search_text_lsa


def five_most_similar_docs(lsa_matrix,search_lsa,clean_mongo_dataframe):
    skl_cso_sim = 1 - cosine_similarity(lsa_matrix,search_lsa).ravel()
    return clean_mongo_dataframe.iloc[skl_cso_sim.argsort()[:5]], clean_mongo_dataframe.iloc[skl_cso_sim.argsort()[0]]