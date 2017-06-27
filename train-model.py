import pymongo
import pickle
import redis

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from wikipedia_helper import get_clean_mongo_dataframe, load_wiki_docs, \
                             fit_svd_for_lsa, preprocess_search_text

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i','--hostip',type=str,help='host IP for MongoDB/Redis')    

args = parser.parse_args()

# Connect to mongo database using host IP argument
mgclient = pymongo.MongoClient(args.hostip)

print("\nRetrieving article texts...")

# Instantiate mongo cursor to pull the collected articles and create a dataframe
curs = mgclient.wikipedia_db.wiki_collector.find()
article_contents_df = get_clean_mongo_dataframe(load_wiki_docs(curs,mgclient.wikipedia_db.wiki_collector.count()))

# Pass the article text through a preprocessing and modeling pipeline and gridsearch hyperparameters for best model
pipe = Pipeline([('tfidf',TfidfVectorizer(stop_words = 'english')),
               ('svd', TruncatedSVD(random_state=12)),
               ('knn', KNeighborsClassifier())])

params = {
            'tfidf__min_df' : [1,2,3],
            'svd__n_components' : [10,50,100],
            'knn__n_neighbors' : range(3,20,2)
}

print("\nFitting the model now...")

gs_pipe = GridSearchCV(pipe, param_grid=params,cv=StratifiedShuffleSplit(random_state=12))
gs_pipe.fit(article_contents_df['clean_text'],article_contents_df['category'])

# Store best fit model and pipeline steps in Redis
r = redis.StrictRedis(args.hostip)
best_models = pickle.dumps(gs_pipe.best_estimator_)
lsa_vectorizer = pickle.dumps(gs_pipe.best_estimator_.steps[0][1])
fit_svd = pickle.dumps(gs_pipe.best_estimator_.steps[1][1])
knn_model = pickle.dumps(gs_pipe.best_estimator_.steps[2][1])
r.set('wiki_best_model',best_models)
r.set('wiki_vectorizer',lsa_vectorizer)
r.set('wiki_fit_svd',fit_svd)
r.set('wiki_knn_model',knn_model)

print("\nThe model is ready to make predictions!")