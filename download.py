import pymongo
import redis
import pickle


from wikipedia_helper import execute_json_query, get_subcategories, \
                             get_article_category_pageid_titles, articles_to_collector, \
                             json_to_dataframe
        
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i','--hostip',type=str,help='host IP for MongoDB/Redis')
parser.add_argument('-c','--categories',type=str,help='Categories for article collector')
parser.add_argument('-s','--subcategories',action='store_true',help='Get articles from subcategories')

args = parser.parse_args()

# Connect to mongoDB using host IP argument and instantiate a database and collection
mgclient = pymongo.MongoClient(args.hostip)
wikipedia_db = mgclient.database_names()
wiki_collector = mgclient.wikipedia_db.wiki_collector

# If the collection already exists, drop the collection and reinstatiate it
if mgclient.wikipedia_db.wiki_collector.count() != 0:
    mgclient.wikipedia_db.drop_collection(wiki_collector)
    wiki_collector = mgclient.wikipedia_db.wiki_collector

# Strip the category name from the urls
wiki_categories = [line.rstrip('\n') for line in open(args.categories)]
wiki_categories = [line.split(':')[2] for line in wiki_categories]

# If the subcategories option is included, pull article contents for categories and subcategoriee
# else only pull article contents for categories  
if args.subcategories:
    pageid_title_df, subcat_dict, uncollected_articles = get_article_category_pageid_titles(wiki_categories,wiki_collector,append_subcats=True)
else:
    pageid_title_df, subcat_dict, uncollected_articles = get_article_category_pageid_titles(wiki_categories,wiki_collector)

# Store the category-subcategory dictionary, pageid and titles dataframe, and list of uncollected article pageids as a validation set 
r = redis.StrictRedis(args.hostip)
subcat_dict = pickle.dumps(subcat_dict)
pageid_title_df = pickle.dumps(pageid_title_df)
uncollected_articles = pickle.dumps(uncollected_articles)
r.set('mlbi_subcat_dict',subcat_dict)
r.set('pageid_title_df',pageid_title_df)
r.set('mlbi_uncollected_articles',uncollected_articles)