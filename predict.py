import pymongo
import redis
import pickle

from wikipedia_helper import preprocess_search_text

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i','--hostip',type=str,help='host IP for MongoDB/Redis')
parser.add_argument('-p','--predict',type=str,help='Text to predict')

args = parser.parse_args() 

# Connect to mongo and redis using the host IP argument
mgclient = pymongo.MongoClient(args.hostip)
r = redis.StrictRedis(args.hostip)

# Load fit models from Redis
lsa_vectorizer = pickle.loads(r.get('wiki_vectorizer'))
fit_svd = pickle.loads(r.get('wiki_fit_svd'))
knn_model = pickle.loads(r.get('wiki_knn_model'))
pageid_title_df = pickle.loads(r.get('pageid_title_df'))

# Preprocess new article text for prediction
predict_lsa = preprocess_search_text(args.predict,lsa_vectorizer,fit_svd)

# Predict the category of the new text and calculate confidence score of prediction
print("\nPredicted Category:", knn_model.predict(predict_lsa)[0])
print("Confidence Score:", knn_model.predict_proba(predict_lsa).max())
