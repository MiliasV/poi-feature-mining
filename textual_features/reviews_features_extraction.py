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
import matplotlib.pyplot as plt


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
    if lang=="en":
        stop = set(stopwords.words('english'))
    elif lang=="nl":
        stop = set(stopwords.words('dutch'))
    elif lang=="el":
        stop = set(stopwords.words("greek"))
    else:
        stop = None
    exclude = set(string.punctuation)
    exclude.add('…')
    return  stop, exclude


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


def get_processed_lda_reviews_from_db(table, city, load):
    if load:
        eng_rev_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/en_rev_text_" + city)
        if city == "ams":
            nl_rev_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/nl_rev_text")
            return eng_rev_text, nl_rev_text

        elif city == "ath":
            el_rev_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/el_rev_text")
            return eng_rev_text, el_rev_text
    else:
        eng_rev = postgis_functions.get_lda_text_per_lang(table, "en")
        eng_rev_text = [t["processedtextlda"].split() for t in eng_rev]
        save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "en_rev_text_" + city, eng_rev_text)
        if city == "ams":
            nl_rev = postgis_functions.get_lda_text_per_lang(table, "nl")
            nl_rev_text = [t["processedtextlda"].split() for t in nl_rev]
            save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                             "nl_rev_text", nl_rev_text)
            return eng_rev_text, nl_rev_text
        elif city == "ath":
            el_rev = postgis_functions.get_lda_text_per_lang(table, "el")
            el_rev_text = [t["processedtextlda"].split() for t in el_rev]
            save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                             "el_rev_text", el_rev_text)
            return eng_rev_text, el_rev_text


def store_google_reviews_and_processed_text(gtable, target_table, city):
    #"matched_google_ams"
    #"matched_google_reviews_ams"
    gpoints = postgis_functions.get_rows_from_table(gtable)
    session, RTable = pois_storing_functions.setup_db(target_table, "notused", "reviews")
    eng_stop, eng_exclude = setup_for_topic_modeling("en")

    if city == "ams":
        nl_stop, nl_exclude = setup_for_topic_modeling("nl")
    elif city == "ath":
        gr_stop, gr_exclude = setup_for_topic_modeling("el")
    lemma = WordNetLemmatizer()

    for g in gpoints:
        rev = {}
        ratings = []
        gjson = json.loads(g["json"])
        if "reviews" in gjson:
            for review in gjson["reviews"]:
                rev["id"] = g["id"] + "_" + review["author_name"]
                rev["placesid"] = g["id"]
                rev["name"] = g["name"]
                rev["type"] = g["type"]
                rev["point"] = g["point"]
                rev["lang"] = review["language"]
                print(rev["lang"])
                rev["text"] = review["text"]
                if rev["lang"] == "en":
                    rev["processedtextlda"] = clean(review["text"], eng_stop, eng_exclude, lemma)
                elif rev["lang"] == "nl" and city=="ams":
                    rev["processedtextlda"] = clean(review["text"], nl_stop, nl_exclude, lemma)
                elif rev["lang"] == "el" and city == "ath":
                    rev["processedtextlda"] = clean(review["text"], gr_stop, gr_exclude, lemma)

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
    per = lda_model.log_perplexity(corpus)
    print('\nPerplexity ', lang, " ",  ntopics , ":",  per)  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score', lang, " ", ntopics, ': ', coherence_lda)
    return coherence_lda, per


def train_lda_models(eng_rev, other_rev, ntopics, passes):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    eng_dict = corpora.Dictionary(eng_rev)
    other_dict = corpora.Dictionary(other_rev)
    # Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    eng_term_matrix = [eng_dict.doc2bow(doc) for doc in eng_rev]
    other_term_matrix = [other_dict.doc2bow(doc) for doc in other_rev]
    corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                               "en_term_matrix_rev_" + city +".mm", eng_term_matrix)
    if city == "ams":
        corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                   "nl_term_matrix_rev.mm", other_term_matrix)
    elif city == "ath":
        corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                   "el_term_matrix_rev.mm", other_term_matrix)

    ldamodel_eng = Lda(eng_term_matrix, num_topics=ntopics, id2word=eng_dict, passes=passes)
    ldamodel_other = Lda(other_term_matrix, num_topics=ntopics, id2word=other_dict, passes=passes)
    return ldamodel_eng, ldamodel_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix


def get_lda_models(eng_rev, other_rev, ntopics, passes, load, evaluate, city):
    if load:
        eng_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                           "en_dict_rev_" + city + ".pkl")
        lda_eng = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                       "lda_en_rev" + "_" + str(ntopics) + "_" + city + ".model")
        eng_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                           "en_term_matrix_rev_" + city + ".mm")
        if city == "ams":
            other_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                              "nl_dict_rev.pkl")
            lda_other = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                          "lda_nl_rev" + "_" + str(ntopics) + ".model")
            other_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                              "nl_term_matrix_rev.mm")
        elif city == "ath":
            other_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                              "el_dict_rev.pkl")
            lda_other = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                          "lda_el_rev" + "_" + str(ntopics) + ".model")
            other_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                                              "el_term_matrix_rev.mm")
    else:
        # Train the LDA model
        if city == "ams":
            lda_eng, lda_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix = train_lda_models(eng_rev, other_rev, ntopics, passes)
            lda_eng.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "lda_en_rev" + "_" + str(ntopics) + "_" + city + ".model")
            lda_other.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                        "lda_nl_rev" + "_" + str(ntopics) + ".model")
            eng_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                          "en_dict_rev_" + city + ".pkl")
            other_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "nl_dict_rev.pkl")
        elif city == "ath":
            lda_eng, lda_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix = train_lda_models(eng_rev, other_rev, ntopics, passes)
            lda_eng.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "lda_en_rev" + "_" + str(ntopics) + "_" + city +".model")
            lda_other.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                        "lda_el_rev" + "_" + str(ntopics) + ".model")
            eng_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                          "en_dict_rev_"+city+".pkl")
            other_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "el_dict_rev.pkl")
    if evaluate:
        coh, per = get_perplexity_and_coherence_score(ntopics, "eng", eng_term_matrix, eng_rev, eng_dict, lda_eng)
        # if city == "ams":
        #     get_perplexity_and_coherence_score(ntopics, "nl", other_term_matrix, other_rev, other_dict, lda_other)
        # elif city == "ath":
        #     get_perplexity_and_coherence_score(ntopics, "el", other_term_matrix, other_rev, other_dict, lda_other)

    return lda_eng, lda_other, eng_dict, other_dict#, coh, per


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
    city = "ams"
    if city == "ams":
        lan = "nl"
    else:
        lan = "el"
    gtable = "matched_places_google_" + city
    rev_table = "matched_places_google_reviews_" + city
    # Store reviews to table and process text for lda
    #store_google_reviews_and_processed_text(gtable, rev_table, city)
    store_table = "matched_places_review_features_10_25_" + city + "_2"
    num_topics_small = 10
    num_topics_big = 25
    session, RTable = pois_storing_functions.setup_db_text(store_table, "reviews",
                                                            num_topics_small = num_topics_small,
                                                            num_topics_big = num_topics_big, lan=lan)
    gpoints = postgis_functions.get_rows_from_table(gtable)
    print("LOADING Reviews....")
    eng_rev, other_rev = get_processed_lda_reviews_from_db(rev_table, city, load=True)
    print("TRAINING Model...")
    # train or load models
    coh = []
    per = []
    top = []
    # for i in [2,4,6,8,10,15,20,25,40]:
    #     print(i)
    #     lda_eng_5, lda_other_5, eng_dict, other_dict, coh_v, per_v = get_lda_models(eng_rev, other_rev,
    #                                                         ntopics=i, passes=20, load=False, evaluate=False,city=city)
    #     coh.append(coh_v)
    #     per.append(per_v)

    lda_eng_10, lda_other_10, eng_dict, other_dict = get_lda_models(eng_rev, other_rev,
                                                        ntopics=num_topics_big, passes=20, load=True, evaluate=False, city=city)

    lda_eng_5, lda_other_5, eng_dict, other_dict = get_lda_models(eng_rev, other_rev,
                                                                                ntopics=num_topics_small, passes=20, load=True,
                                                                                evaluate=False, city=city)
    # , coh_v, per_v
    # print(lda_eng_5.show_topics(num_topics=10, num_words=5))
    print(lda_other_5.show_topics(num_topics=10, num_words=5))
    # print(lda_eng_10.show_topics(num_topics=25, num_words=5))
    print(a)
    for g in gpoints:
        print(g)
        data = {}
        data["id"] = g["id"]
        data["name"] = g["name"]
        data["placesid"] = g["id"]
        data["lat"] = g["lat"]
        data["lng"] = g["lng"]
        data["type"] = g["type"]

        ######################
        # LDA Topic Modeling #
        ######################
        #  get tweets per place per lang
        eng_rev_lda = postgis_functions.get_col_from_feature_per_lang(rev_table, "processedtextlda","placesid", g["id"], "en")
        if city == "ams":
            other_rev_lda = postgis_functions.get_col_from_feature_per_lang(rev_table ,
                                                                         "processedtextlda", "placesid", g["id"], "nl")
        elif city == "ath":
            other_rev_lda = postgis_functions.get_col_from_feature_per_lang(rev_table,
                                                                         "processedtextlda", "placesid", g["id"], "el")

        print("GETTING Topics from ", len(eng_rev_lda) + len(other_rev_lda), " Reviews: Eng = ",
              len(eng_rev_lda) , ", OTHER = ", len(other_rev_lda))


        eng_topics_5 = get_topics_from_lda(eng_rev_lda, lda_eng_5, eng_dict, num_topics=num_topics_small)
        other_topics_5 = get_topics_from_lda(other_rev_lda, lda_other_5, other_dict, num_topics=num_topics_small)

        eng_topics_10 = get_topics_from_lda(eng_rev_lda, lda_eng_10, eng_dict, num_topics=num_topics_big)
        other_topics_10 = get_topics_from_lda(other_rev_lda, lda_other_10, other_dict, num_topics=num_topics_big)

        for i, val in enumerate(eng_topics_5):
            data["topiceng" + str(num_topics_small) + str(i+1)] = float(eng_topics_5[i])
            data["topic" + lan + str(num_topics_small) + str(i+1)] = float(other_topics_5[i])
        for i, val in enumerate(eng_topics_10):
            data["topiceng" + str(num_topics_big) + str(i+1)] = float(eng_topics_10[i])
            data["topic" + lan + str(num_topics_big) + str(i + 1)] = float(other_topics_10[i])


        ####################
        # Review statistics #
        ####################
        # get unprocessed text
        eng_rev = postgis_functions.get_col_from_feature_per_lang(rev_table, "text", "placesid", g["id"], "en")
        if city == "ams":
            other_rev = postgis_functions.get_col_from_feature_per_lang(rev_table, "text", "placesid", g["id"], "nl")
        elif city == "ath":
            other_rev = postgis_functions.get_col_from_feature_per_lang(rev_table, "text", "placesid", g["id"], "el")

        # count of reviews

        data["enrevcount"] = len(eng_rev)
        data[lan + "revcount"] = len(other_rev)
        data["totalrevcount"] = data["enrevcount"] + data[lan + "revcount"]

        # make reviews a list of tweets
        eng_rev = [x["text"] for x in eng_rev if x["text"]!=""]
        other_rev = [x["text"] for x in other_rev if x["text"]!=""]

        # count of words
        data["enwordcount"] = sum([len(x.split(" ")) for x in eng_rev])
        data[lan + "wordcount"] = sum([len(x.split(" ")) for x in other_rev])
        data["totalwordcount"] = data["enwordcount"] + data[lan + "wordcount"]

        # avg. count of words
        if data["enrevcount"]!=0:
            data["engavgword"] = data["enwordcount"] / data["enrevcount"]
        else:
            data["engavgword"] = 0
        if data[lan + "revcount"]!=0:
            data[lan + "avgword"] = data[lan + "wordcount"] / data[lan + "revcount"]
        else:
            data[lan + "avgword"] = 0
        data["avgword"] = (data["engavgword"] + data[lan + "avgword"]) / 2.0


        ######################
        # Sentiment analysis #
        ######################
        # POLYGLOT
        data["enpolpoly"] = tweet_features_extraction.get_sent_from_polyglot(eng_rev, "en")
        if city == "ams":
            data["nlpolpoly"] = tweet_features_extraction.get_sent_from_polyglot(other_rev, "nl")
        elif city == "ath":
            data["elpolpoly"] = tweet_features_extraction.get_sent_from_polyglot(other_rev, "el")
        ############
        # TextBlob #
        ############
        # for english reviews
        data["enpolblob"], data["ensubjblob"] = tweet_features_extraction.get_sent_from_textblob(eng_rev)
        # for dutch tweets translate first
        if other_rev:
            other_trans_rev = tweet_features_extraction.get_text_trans_in_eng(other_rev)
            data[lan + "polblob"], data[lan + "subblob"] = tweet_features_extraction.get_sent_from_textblob(other_trans_rev)
        else:
            data[lan + "polblob"], data[lan + "subblob"] = None, None
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
