import re
import pymongo

from wikipedia_helper import get_clean_mongo_dataframe, load_wiki_docs, \
                             fit_svd_for_lsa, preprocess_search_text, \
                             five_most_similar_docs

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i','--hostip',type=str,help='host IP for MongoDB/Redis')   
parser.add_argument('-q','--query',type=str,help='Text to query')

args = parser.parse_args()

# Connect to mongo database using host IP argument
mgclient = pymongo.MongoClient(args.hostip)
curs = mgclient.wikipedia_db.wiki_collector.find()
article_contents_df = get_clean_mongo_dataframe(load_wiki_docs(curs,mgclient.wikipedia_db.wiki_collector.count()))

# Perform tfidf and svd on articles to train search model
vectorizer, fitted_svd, lsa = fit_svd_for_lsa(article_contents_df['clean_text'])

args = parser.parse_args()

# Preprocess search query using fit tfidf and svd
search_lsa = preprocess_search_text(args.query,vectorizer,fitted_svd)

# Perform search to find the 5 closest articles and the website and first paragragh of the closest article 
top_five, top_text = five_most_similar_docs(lsa,search_lsa,article_contents_df)

print("\n")
print("The top 5 related articles are the following:\n")
print(top_five[['category','title']],"\n\n")

print("The text from the top related article is below:\n")
print("Title:", top_text[['title']][0],"\n")
print("Website: https://en.wikipedia.org/wiki/{}".format(re.sub('\s','_',top_text[['title']][0])),"\n")
print("Introduction:", top_text[['clean_text']][0].split('\n',1)[0])                            