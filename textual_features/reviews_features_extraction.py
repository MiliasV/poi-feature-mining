import json
import pprint
import postgis_functions
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import string
import re
import emoji
import pois_storing_functions
import pickle
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import sys

from textual_features import tweet_features_extraction


def load_with_pickle(filename):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def save_with_pickle(filename, obj):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'wb') as f:
        pickle.dump(obj, f)


def setup_for_topic_modeling(lang):
    # initialize lemmatizer
    lemma = WordNetLemmatizer()
    if lang=="en":
        stop = set(stopwords.words('english'))
    elif lang=="nl":
        stop = set(stopwords.words('dutch'))
    else:
        stop = None
    exclude = set(string.punctuation)
    exclude.add('…')
    return lemma, stop, exclude


def clean(doc, stop, exclude, lemma):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    res = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    res = res.replace("amsterdam", "")
    #remove links
    res = re.sub(r"http\S+", "", res)
    #remove numbers
    res = re.sub(r"\d", "", res)
    # remove words with len<=2
    res = re.sub(r'\b\w{1,2}\b', '', res)
    # emojis to text
    res = emoji.demojize(res)
    # remove ":"
    res = res.replace(":", " ")
    return res


def get_processed_lda_reviews_from_db(table, load):
    if load:
        eng_rev_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                        "en_rev_text")
        nl_rev_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                       "nl_rev_text")
        return eng_rev_text, nl_rev_text
    else:
        eng_rev = postgis_functions.get_lda_text_per_lang(table, "en")
        eng_rev_text = [t["processedtextlda"].split() for t in eng_rev]
        nl_rev = postgis_functions.get_lda_text_per_lang(table, "nl")
        nl_rev_text = [t["processedtextlda"].split() for t in nl_rev]
        # Store reviews
        save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "en_rev_text", eng_rev_text)
        save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "nl_rev_text", nl_rev_text)
        return eng_rev_text, nl_rev_text


def store_google_reviews_and_processed_text():
    gpoints = postgis_functions.get_rows_from_table("matched_google_ams")
    session, RTable = pois_storing_functions.setup_db("matched_google_reviews_ams", "notused", "reviews")
    lemma, eng_stop, eng_exclude = setup_for_topic_modeling("en")
    lemma, nl_stop, nl_exclude = setup_for_topic_modeling("nl")
    for g in gpoints:
        rev = {}
        ratings = []
        gjson = json.loads(g["json"])
        if "reviews" in gjson:
            for review in gjson["reviews"]:
                rev["id"] = g["id"] + "_" + review["author_name"]
                rev["gid"] = g["id"]
                rev["name"] = g["name"]
                rev["type"] = g["type"]
                rev["point"] = g["point"]
                rev["lang"] = review["language"]
                rev["text"] = review["text"]
                if rev["lang"] == "en":
                    rev["processedldatext"] = clean(review["text"], eng_stop, eng_exclude, lemma)
                elif rev["lang"] == "nl":
                    rev["processedldatext"] = clean(review["text"], nl_stop, nl_exclude, lemma)

                # ratings.append(review["rating"])
                # rev["avgrating"] = np.mean(ratings)
                print("############################################################")
                try:
                    session.add(RTable(**rev))
                    session.commit()
                    print(rev["name"], " INSERTED!")
                except Exception as err:
                    session.rollback()
                    print("# NOT INSERTED: ", err)
                print("############################################################")


def get_perplexity_and_coherence_score(ntopics, lang, corpus,
                                       doc, dict, lda_model):
    ################################################################3
    # checking the models!
    # Compute Perplexity
    print('\nPerplexity (eng , ', ntopics, ': ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score', ntopics, lang, ': ', coherence_lda)


def train_lda_models(eng_rev, nl_rev, ntopics, passes):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    eng_dict = corpora.Dictionary(eng_rev)
    nl_dict = corpora.Dictionary(nl_rev)
    # Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    eng_term_matrix = [eng_dict.doc2bow(doc) for doc in eng_rev]
    nl_term_matrix = [nl_dict.doc2bow(doc) for doc in nl_rev]
    corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                               "en_term_matrix_rev.mm", eng_term_matrix)
    corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                               "nl_term_matrix_rev.mm", nl_term_matrix)
    ldamodel_eng = Lda(eng_term_matrix, num_topics=ntopics, id2word=eng_dict, passes=passes)
    ldamodel_nl = Lda(nl_term_matrix, num_topics=ntopics, id2word=nl_dict, passes=passes)
    return ldamodel_eng, ldamodel_nl, eng_dict, nl_dict, eng_term_matrix, nl_term_matrix


def get_lda_models(eng_rev, nl_rev, ntopics, passes, load, evaluate):
    if load:
        eng_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                           "en_dict_rev.pkl")
        nl_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                          "nl_dict_rev.pkl")
        lda_eng = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                       "lda_en_rev" + "_" + str(ntopics) + ".model")
        lda_nl = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                      "lda_nl_rev" + "_" + str(ntopics) + ".model")
        eng_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                           "en_term_matrix_rev.mm")
        nl_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                          "nl_term_matrix_rev.mm")
    else:
        # Train the LDA model
        lda_eng, lda_nl, eng_dict, nl_dict, eng_term_matrix, nl_term_matrix = train_lda_models(eng_rev, nl_rev, ntopics, passes)
        lda_eng.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                     "lda_en_rev" + "_" + str(ntopics) + ".model")
        lda_nl.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                    "lda_nl_rev" + "_" + str(ntopics) + ".model")
        eng_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                      "en_dict_rev.pkl")
        nl_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                     "nl_dict_rev.pkl")
    if evaluate:
        get_perplexity_and_coherence_score(ntopics, "eng", eng_term_matrix, eng_rev, eng_dict, lda_eng)
        get_perplexity_and_coherence_score(ntopics, "nl", nl_term_matrix, nl_rev, nl_dict, lda_nl)
    return lda_eng, lda_nl, eng_dict, nl_dict


def get_topics_from_lda(obj, model, dict, num_topics):
    topics = []
    place_topics = []
    if obj:
        obj_text = [t["processedtextlda"].split() if t["processedtextlda"] is not None else [''] for t in obj]
        #print(tweets_text)
        #eng_tweets_text = [t for tweet in eng_tweets_text for t in tweet]
        for t in obj_text:
            #print(t)
            bow = dict.doc2bow(t)
            topic = model.get_document_topics(bow)
            # if not all topics discovered add them with prob 0
            if len(topic)<num_topics:
                for i in range(num_topics):
                    if True not in [i in x for x in topic]:
                        topic.append((i,0.0))
            # sort topics by topic class to take the mean afterwards
            topic.sort(key=lambda tup: tup[0])  # sorts in place
            topics.append(topic)
        for i in range(num_topics):
            place_topics.append(np.mean([item[i][1] for item in topics]))
    else:
        place_topics = [-1] * num_topics
    return place_topics


if __name__ == '__main__':
    # Store reviews to table and process text for lda
    # store_google_reviews_and_processed_text()
    gpoints = postgis_functions.get_rows_from_table("matched_google_ams")
    session, RTable = pois_storing_functions.setup_db("matched_review_features_ams", "notused", "review_features")

    print("LOADING Reviews....")
    num_topics = 5
    eng_rev, nl_rev = get_processed_lda_reviews_from_db("matched_google_reviews_ams", load=False)
    print("TRAINING Model...")

    # train or load models
    lda_eng_5, lda_nl_5, eng_dict, nl_dict = get_lda_models(eng_rev, nl_rev,
                                                        ntopics=num_topics, passes=20, load=False, evaluate=False)
    lda_eng_10, lda_nl_10, eng_dict, nl_dict = get_lda_models(eng_rev, nl_rev,
                                                        ntopics=10, passes=20, load=False, evaluate=False)

    for g in gpoints:
        print(g)
        data = {}
        data["id"] = g["id"]
        data["name"] = g["name"]
        data["point"] = g["point"]
        data["lat"] = g["lat"]
        data["lng"] = g["lng"]
        data["type"] = g["type"]

        ######################
        # LDA Topic Modeling #
        ######################
        #  get tweets per place per lang
        eng_rev_lda = postgis_functions.get_col_from_feature_per_lang("matched_google_reviews_ams",
                                                                      "processedtextlda","gid", g["id"], "en")
        nl_rev_lda = postgis_functions.get_col_from_feature_per_lang("matched_google_reviews_ams",
                                                                "processedtextlda", "gid", g["id"], "nl")

        print("GETTING Topics from ", len(eng_rev_lda) + len(nl_rev_lda), " Reviews: Eng = ",
              len(eng_rev_lda) , ", NL = ", len(nl_rev_lda))
        #print(lda_eng_5.show_topics(num_topics=5, num_words=5))
        #print(lda_eng_10.show_topics(num_topics=10, num_words=5))

        eng_topics_5 = get_topics_from_lda(eng_rev_lda, lda_eng_5, eng_dict, num_topics=5)
        nl_topics_5 = get_topics_from_lda(nl_rev_lda, lda_nl_5, nl_dict, num_topics=5)

        eng_topics_10 = get_topics_from_lda(eng_rev_lda, lda_eng_10, eng_dict, num_topics=10)
        nl_topics_10 = get_topics_from_lda(nl_rev_lda, lda_nl_10, nl_dict, num_topics=10)

        for i, val in enumerate(eng_topics_5):
            data["topiceng5" + str(i+1)] = float(eng_topics_5[i])
            data["topicnl5" + str(i+1)] = float(nl_topics_5[i])
        for i, val in enumerate(eng_topics_10):
            data["topiceng10" + str(i+1)] = float(eng_topics_10[i])
            data["topicnl10" + str(i + 1)] = float(nl_topics_10[i])


        ####################
        # Review statistics #
        ####################
        # get unprocessed text
        eng_rev = postgis_functions.get_col_from_feature_per_lang("matched_google_reviews_ams", "text", "gid", g["id"], "en")
        nl_rev = postgis_functions.get_col_from_feature_per_lang("matched_google_reviews_ams", "text", "gid", g["id"], "nl")
        # count of reviews
        data["enrevcount"] = len(eng_rev)
        data["nlrevcount"] = len(nl_rev)
        data["totalrevcount"] = data["enrevcount"] + data["nlrevcount"]

        # make reviews a list of tweets
        eng_rev = [x["text"] for x in eng_rev if x["text"]!=""]
        nl_rev = [x["text"] for x in nl_rev if x["text"]!=""]

        # count of words
        data["enwordcount"] = sum([len(x.split(" ")) for x in eng_rev])
        data["nlwordcount"] = sum([len(x.split(" ")) for x in nl_rev])
        data["totalwordcount"] = data["enwordcount"] + data["nlwordcount"]

        # avg. count of words
        if data["enrevcount"]!=0:
            data["engavgword"] = data["enwordcount"] / data["enrevcount"]
        else:
            data["engavgword"] = 0
        if data["nlrevcount"]!=0:
            data["nlavgword"] = data["nlwordcount"] / data["nlrevcount"]
        else:
            data["nlavgword"] = 0
        data["avgword"] = (data["engavgword"] + data["nlavgword"]) / 2.0


        ######################
        # Sentiment analysis #
        ######################
        # POLYGLOT
        data["enpolpoly"] = tweet_features_extraction.get_sent_from_polyglot(eng_rev, "en")
        data["nlpolpoly"] = tweet_features_extraction.get_sent_from_polyglot(nl_rev, "nl")

        ############
        # TextBlob #
        ############
        # for english reviews
        data["enpolblob"], data["ensubjblob"] = tweet_features_extraction.get_sent_from_textblob(eng_rev)
        # for dutch tweets translate first
        if nl_rev:
            nl_trans_rev = tweet_features_extraction.get_text_trans_in_eng(nl_rev)
            data["nlpolblob"], data["nlsubblob"] = tweet_features_extraction.get_sent_from_textblob(nl_trans_rev)
        else:
            data["nlpolblob"], data["nlsubblob"] = None, None
        print("############################################################")
        #############
        # Add to db #
        #############
        try:
            session.add(RTable(**data))
            session.commit()
            print(data["name"], " INSERTED!")
        except Exception as err:
            session.rollback()
            print("# NOT INSERTED: ", err)
        print("############################################################")
