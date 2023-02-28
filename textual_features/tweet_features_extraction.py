from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import langid
import re
import postgis_functions
# Importing Gensim
import gensim
from gensim import corpora, models
import numpy as np
import re
import pprint
import sys
import json
sys.path.append("..")
sys.path.append("../..")
import emoji
import pickle
import ast
from datetime import datetime
from datetime import timedelta
import pois_storing_functions
import time
from gensim.models import CoherenceModel
import polyglot
from polyglot.text import Text
from textblob import TextBlob


try:
    import tweet2vec
    import emoji
    from textblob import TextBlob
    from polyglot.downloader import downloader
    from polyglot.text import Text
    from googletrans import Translator
except:
    print("RUNNING FOR Python 2.7")


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


def get_text_language(text):
    # get text's language
    lang = langid.classify(text)[0]
    return lang


def change_language_polyglot_list(poly, lang):
    for obj in poly:
        obj.language = lang
    return poly


def get_sent_from_polyglot(tweets, lan):
    t = [Text(t) for t in tweets]
    t = change_language_polyglot_list(t, lan)
    t_sent = np.mean([tweet.polarity for tweet in t])
    # try:
    #
    # except:
    #     t_sent = None
    return t_sent


def get_sent_from_textblob(tweets):
    tb = [TextBlob(emoji.demojize(t)) for t in tweets]
    return np.mean([t.polarity for t in tb]) , np.mean([t.subjectivity for t in tb])


def get_text_trans_in_eng(text_list):
    # convert emojis to string and then translate
    try:
        return [Translator().translate(emoji.demojize(tweet)).text for tweet in text_list
                if emoji.demojize(tweet) is not None]
    except:
        time.sleep(5)
        tlist = text_list.copy()
        for i in tlist:
            try:
                Translator().translate(emoji.demojize(i))
            except:
                text_list.remove(i)
        print(tlist)
        print(text_list)
        return [Translator().translate(emoji.demojize(tweet)).text for tweet in text_list
                if (emoji.demojize(tweet) is not None)]


def get_perplexity_and_coherence_score(ntopics, lang, corpus,
                                       doc, dict, lda_model):
    ################################################################3
    # checking the models!
    # Compute Perplexity
    print('\nPerplexity  , ' , lang ," ", ntopics, ': ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc, dictionary=dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score', ntopics, lang, ': ', coherence_lda)


def train_lda_models(eng_tweets, other_tweets, ntopics, passes):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    eng_dict = corpora.Dictionary(eng_tweets)
    other_dict = corpora.Dictionary(other_tweets)
    # Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    eng_term_matrix = [eng_dict.doc2bow(doc) for doc in eng_tweets]
    other_term_matrix = [other_dict.doc2bow(doc) for doc in other_tweets]
    corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                               "objects/en_term_matrix_tweets_" +city + ".mm", eng_term_matrix)
    if city == "ams":
        corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                               "objects/nl_term_matrix_tweets.mm", other_term_matrix)
    elif city == "ath":
        corpora.MmCorpus.serialize("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                               "objects/el_term_matrix_tweets.mm", other_term_matrix)
    ldamodel_eng = Lda(eng_term_matrix, num_topics=ntopics, id2word=eng_dict, passes=passes)
    ldamodel_other = Lda(other_term_matrix, num_topics=ntopics, id2word=other_dict, passes=passes)
    return ldamodel_eng, ldamodel_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix


def get_lda_models(eng_tweets, other_tweets, ntopics, passes, load, evaluate, city):
    if load:
        eng_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                           "objects/en_dict_tweets_" + city + ".pkl")
        lda_eng = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                       "objects/lda_en_tweets" + "_" + str(ntopics) + "_" + city + ".model")
        eng_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                           "objects/en_term_matrix_tweets_" + city + ".mm")
        if city=="ams":
            other_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                              "objects/nl_dict_tweets.pkl")
            lda_other = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                          "objects/lda_nl_tweets" + "_" + str(ntopics) + ".model")
            other_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                              "objects/nl_term_matrix_tweets.mm")
        elif city == "ath":
            other_dict = corpora.Dictionary.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                                 "objects/el_dict_tweets.pkl")
            lda_other = models.LdaModel.load("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                             "objects/lda_el_tweets" + "_" + str(ntopics) + ".model")
            other_term_matrix = corpora.MmCorpus("/home/bill/Desktop/thesis/code/UDS/textual_features/"
                                                 "objects/el_term_matrix_tweets.mm")
    else:
        # Train the LDA model
        if city =="ams":
            lda_eng, lda_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix = train_lda_models(eng_tweets, other_tweets, ntopics, passes)
            lda_eng.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "lda_en_tweets" + "_" + str(ntopics) + "_" + city +  ".model")
            lda_other.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                        "lda_nl_tweets" + "_" + str(ntopics) + ".model")
            eng_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                          "en_dict_tweets_" + city + ".pkl")
            other_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "nl_dict_tweets.pkl")
        if city =="ath":
            lda_eng, lda_other, eng_dict, other_dict, eng_term_matrix, other_term_matrix = train_lda_models(eng_tweets, other_tweets, ntopics, passes)
            lda_eng.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "lda_en_tweets" + "_" + str(ntopics) + "_" + city + ".model")
            lda_other.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                        "lda_el_tweets" + "_" + str(ntopics) + ".model")
            eng_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                          "en_dict_tweets_" + city + ".pkl")
            other_dict.save("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "el_dict_tweets.pkl")

    if evaluate:
        get_perplexity_and_coherence_score(ntopics, "eng", eng_term_matrix, eng_tweets, eng_dict, lda_eng)
        if city =="ams":
            get_perplexity_and_coherence_score(ntopics, "nl", other_term_matrix, other_tweets, other_dict, lda_other)
        elif city =="ath":
            get_perplexity_and_coherence_score(ntopics, "el", other_term_matrix, other_tweets, other_dict, lda_other)

    return lda_eng, lda_other, eng_dict, other_dict


def setup_for_topic_modeling(lang):
    # initialize lemmatizer
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
    return stop, exclude

#
# def get_text_tweets(table,  eng_stop, nl_stop, eng_exclude, nl_exclude, lemma):
#     eng_tweets = []
#     nl_tweets = []
#     count=0
#     tweets = postgis_functions.get_rows_from_table(table)
#     for t in tweets:
#         #count+=1
#         #print(count)
#         lang = get_text_language(t["text"])
#         if lang == "en":
#             # remove stopwords, punctuation, links and lemmatize words
#             eng_tweets.append(clean(t["text"], eng_stop, eng_exclude, lemma).split())
#         elif lang == "nl":
#             # remove stopwords, punctuation, links and lemmatize words
#             nl_tweets.append(clean(t["text"], nl_stop, nl_exclude, lemma).split())
#     return eng_tweets, nl_tweets


def add_processed_lda_text_tweets(table, city):
    # Process tweets for LDA-based topic modelling - Load = True uses saved data
    # stop words and punctuations
    eng_stop, eng_exclude = setup_for_topic_modeling("en")
    if city == "ams":
        nl_stop, nl_exclude = setup_for_topic_modeling("nl")
    elif city == "ath":
        gr_stop, gr_exclude = setup_for_topic_modeling("el")
    lemma = WordNetLemmatizer()

    tweets = postgis_functions.get_rows_from_table_where_col_is_null(table, "processedtextlda")
    for t in tweets:
        lang = t["lang"]
        if lang == "en":
            # remove stopwords, punctuation, links and lemmatize words
            postgis_functions.add_processed_text_to_table(clean(t["text"], eng_stop, eng_exclude, lemma),
                                                          "processedtextlda", table, t["id"])
        elif lang == "nl" and city=="ams":
            # remove stopwords, punctuation, links and lemmatize words
            postgis_functions.add_processed_text_to_table(clean(t["text"], nl_stop, nl_exclude, lemma),
                                                          "processedtextlda", table, t["id"])
        elif lang == "el" and city == "ath":
            # remove stopwords, punctuation, links and lemmatize words
            postgis_functions.add_processed_text_to_table(clean(t["text"], gr_stop, gr_exclude, lemma),
                                                          "processedtextlda", table, t["id"])


def get_processed_lda_tweets_from_db(table, city, load):
    if load:
        eng_tweets_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/"
                                           "textual_features/objects/en_tweets_text_" + city)
        if city == "ams":
            nl_tweets_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/"
                                              "textual_features/objects/nl_tweets_text")
            return eng_tweets_text, nl_tweets_text
        elif city == "ath":
            el_tweets_text = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/"
                                              "textual_features/objects/el_tweets_text")
            return eng_tweets_text, el_tweets_text

    else:
        eng_tweets = postgis_functions.get_lda_text_per_lang(table, "en")
        eng_tweets_text = [t["processedtextlda"].split() for t in eng_tweets]
        save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "en_tweets_text_" + city, eng_tweets_text)

        if city == "ams":
            nl_tweets = postgis_functions.get_lda_text_per_lang(table, "nl")
            nl_tweets_text = [t["processedtextlda"].split() for t in nl_tweets]
            save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "nl_tweets_text", nl_tweets_text)
            return eng_tweets_text, nl_tweets_text
        elif city == "ath":
            el_tweets = postgis_functions.get_lda_text_per_lang(table, "el")
            el_tweets_text = [t["processedtextlda"].split() for t in el_tweets]
            save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
                         "el_tweets_text", el_tweets_text)
            return eng_tweets_text, el_tweets_text


# def get_processed_tweets(table, load):
#     if load:
#         eng_tweets = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
#                                       "en_tweets")
#         nl_tweets = load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
#                                      "nl_tweets")
#         return eng_tweets, nl_tweets
#     else:
#         # stop words and punctuations
#         lemma, eng_stop, eng_exclude = setup_for_topic_modeling("en")
#         lemma, nl_stop, nl_exclude = setup_for_topic_modeling("nl")
#         # takes around 5 mins (saved for now)
#         eng_tweets, nl_tweets = get_text_tweets(table, eng_stop, nl_stop,
#                                                 eng_exclude, nl_exclude, lemma)
#         # Store tweets
#         save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
#                          "eng_tweets", eng_tweets)
#         save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
#                          "nl_tweets", nl_tweets)
#         return eng_tweets, nl_tweets


def load_with_pickle(filename):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def save_with_pickle(filename, obj):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'wb') as f:
        pickle.dump(obj, f)


def update_language_from_langid():
    tweets = postgis_functions.get_rows_from_table("matched_twitter_ams")
    for t in tweets:
        lang = get_text_language(t["text"])
        # lang = json.loads(t["json"])["lang"]
        postgis_functions.update_tweets_language("matched_twitter_ams", lang, t["id"])


def get_time_diffs(data, table, id):
    tweetimes = postgis_functions.get_daytime_by_id(table, id)
    # create datetime objects
    tweetimes = [datetime.strptime(x["createdat"], format_str) for x in tweetimes]
    tweetimes_dif = []
    for t1, t2 in zip(tweetimes, tweetimes[1:]):
        tweetimes_dif.append((t2 - t1) / timedelta(days=1))
    data["timediffavg"] = np.mean(tweetimes_dif)
    data["timediffmedian"] = np.median(tweetimes_dif)
    return data


def get_topics_from_lda(tweets, model, dict, num_topics):
    topics = []
    place_topics = []
    if tweets:
        tweets_text = [t["processedtextlda"].split() if t["processedtextlda"] is not None else [''] for t in tweets]
        #print(tweets_text)
        #eng_tweets_text = [t for tweet in eng_tweets_text for t in tweet]
        for t in tweets_text:
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
        place_topics = [0] * num_topics
    return place_topics


def add_matched_placesid_from_fsqid(store_table, ftable, gtable):
    fpoints = postgis_functions.get_rows_from_table(ftable)
    for f in fpoints:
        placesid = postgis_functions.get_matched_placesid_from_fsqid(f["id"], ftable, gtable)
        postgis_functions.update_column_to_table_by_key(store_table, placesid, "placesid", "id", f["id"])
    print("Places ids added !")


if __name__ == '__main__':
    # initializations
    city = "ath"
    if city == "ams":
        lan = "nl"
    else:
        lan = "el"
    ftable = "matched_places_fsq_" + city
    gtable = "matched_places_google_" + city
    tweet_table = "matched_places_twitter_" + city
    store_table = "matched_places_text_features_10_25_" + city + "_2"
    num_topics_small = 10
    num_topics_big = 25
    session, TFTable = pois_storing_functions.setup_db_text(store_table, "twitter",
                                                            num_topics_small = 10,
                                                            num_topics_big = 25, lan=lan)
    format_str = "%Y-%m-%d %H:%M:%S"

    # get places
    fpoints = postgis_functions.get_rows_from_table(ftable)

    #add_matched_placesid_from_fsqid(store_table, ftable, gtable)
    #fpoints = postgis_functions.get_rows_from_id_not_in_table(ftable, store_table, "id")

    # ADD processed tweets for lda to db
    add_processed_lda_text_tweets(tweet_table, city)
    # get processed tweets for lda per language
    print("LOADING Tweets....")
    eng_tweets, other_tweets = get_processed_lda_tweets_from_db(tweet_table, load=True, city=city)
    print("TRAINING Model...")
    # train or load models
    lda_eng_5, lda_other_5, eng_dict, other_dict = get_lda_models(eng_tweets, other_tweets,
                                                        ntopics=num_topics_small, passes=20, load=True, evaluate=False, city=city)

    lda_eng_10, lda_other_10, eng_dict, other_dict = get_lda_models(eng_tweets, other_tweets,
                                                        ntopics=num_topics_big, passes=20, load=True, evaluate=False, city=city)

    print(lda_eng_5.show_topics(num_topics=10, num_words=5))
    print(lda_eng_10.show_topics(num_topics=25, num_words=5))
    print(lda_other_5.show_topics(num_topics=25, num_words=5))
    print(lda_other_10.show_topics(num_topics=25, num_words=5))

    # print(a)
    # for every matched place
    for f in fpoints:
        data = {}
        fpoint = {k: v if v is not None else "" for k, v in f.items()}
        data["id"] = fpoint["id"]
        data["name"] = fpoint["name"]
        data["point"] = fpoint["point"]
        data["lat"] = fpoint["lat"]
        data["lng"] = fpoint["lng"]
        data["type"] = postgis_functions.get_type_of_place(fpoint)
        data["placesid"] = postgis_functions.get_matched_placesid_from_fsqid(fpoint["id"],ftable, gtable)
        print(fpoint)
        data = get_time_diffs(data, tweet_table, fpoint["id"])

        ######################
        # LDA Topic Modeling #
        ######################
        #  get tweets per place per lang
        eng_tweets_lda = postgis_functions.get_col_from_feature_per_lang(tweet_table, "processedtextlda","fsqid", fpoint["id"], "en")

        if city =="ams":
            other_tweets_lda = postgis_functions.get_col_from_feature_per_lang(tweet_table, "processedtextlda", "fsqid",
                                                                             fpoint["id"], "nl")
        elif city =="ath":
            other_tweets_lda = postgis_functions.get_col_from_feature_per_lang(tweet_table, "processedtextlda", "fsqid",
                                                                               fpoint["id"], "el")


        print("GETTING Topics from ", len(eng_tweets_lda) + len(other_tweets_lda), " Reviews: Eng = ",
              len(eng_tweets_lda) , ", OTHER = ", len(other_tweets_lda))
        #print(lda_eng_5.show_topics(num_topics=5, num_words=5))
        #print(lda_eng_10.show_topics(num_topics=10, num_words=5))

        eng_topics_5 = get_topics_from_lda(eng_tweets_lda, lda_eng_5, eng_dict, num_topics=num_topics_small)
        other_topics_5 = get_topics_from_lda(other_tweets_lda, lda_other_5, other_dict, num_topics=num_topics_small)

        eng_topics_10 = get_topics_from_lda(eng_tweets_lda, lda_eng_10, eng_dict, num_topics=num_topics_big)
        other_topics_10 = get_topics_from_lda(other_tweets_lda, lda_other_10, other_dict, num_topics=num_topics_big)

        # for i in range(len(eng_topics)):
        #     print(eng_topics[i], lda_eng.show_topic(i, topn=5))

        for i, val in enumerate(eng_topics_5):
            data["topiceng" + str(num_topics_small) + str(i+1)] = float(eng_topics_5[i])
            data["topic" + lan + str(num_topics_small) + str(i+1)] = float(other_topics_5[i])
        for i, val in enumerate(eng_topics_10):
            data["topiceng" + str(num_topics_big) + str(i+1)] = float(eng_topics_10[i])
            data["topic" + lan + str(num_topics_big) + str(i + 1)] = float(other_topics_10[i])

        ####################
        # Tweet statistics #
        ####################
        # get unprocessed text
        eng_tweets = postgis_functions.get_col_from_feature_per_lang(tweet_table, "text", "fsqid", fpoint["id"], "en")
        if city == "ams":
            other_tweets = postgis_functions.get_col_from_feature_per_lang(tweet_table, "text", "fsqid", fpoint["id"], "nl")
        elif city == "ath":
            other_tweets = postgis_functions.get_col_from_feature_per_lang(tweet_table, "text", "fsqid", fpoint["id"], "el")

        # count of tweets - Think about it as I gather until 50-100 tweets per place
        data["entweetcount"] = len(eng_tweets)
        data[lan + "tweetcount"] = len(other_tweets)
        data["totaltweetcount"] = data["entweetcount"] + data[lan + "tweetcount"]

        # make tweets a list of tweets
        eng_tweets = [x["text"] for x in eng_tweets if x["text"]!=""]
        other_tweets = [x["text"] for x in other_tweets if x["text"]!=""]

        # count of words
        data["enwordcount"] = sum([len(x.split(" ")) for x in eng_tweets])
        data[lan + "wordcount"] = sum([len(x.split(" ")) for x in other_tweets])
        data["totalwordcount"] = data["enwordcount"] + data[lan + "wordcount"]

        # avg. count of words
        if data["entweetcount"]!=0:
            data["engavgword"] = data["enwordcount"] / data["entweetcount"]
        else:
            data["engavgword"] = 0
        if data[lan + "tweetcount"]!=0:
            data[lan + "avgword"] = data[lan + "wordcount"] / data[lan + "tweetcount"]
        else:
            data[lan+ "avgword"] = 0
        data["avgword"] = (data["engavgword"] + data[lan + "avgword"]) / 2.0

        ######################
        # Sentiment analysis #
        ######################
        # POLYGLOT
        data["enpolpoly"] = get_sent_from_polyglot(eng_tweets, "en")
        if city == "ams":
            data["nlpolpoly"] = get_sent_from_polyglot(other_tweets, "nl")
        elif city == "ath":
            data["elpolpoly"] = get_sent_from_polyglot(other_tweets, "el")


        ############
        # TextBlob #
        ############
        # for english tweets
        data["enpolblob"], data["ensubjblob"] = get_sent_from_textblob(eng_tweets)
        # for dutch tweets translate first
        if other_tweets:
            other_trans_tweets = get_text_trans_in_eng(other_tweets)
            data[lan + "polblob"], data[lan + "subblob"] = get_sent_from_textblob(other_trans_tweets)
        else:
            data[lan + "polblob"], data[lan + "subblob"] = None, None
        print("############################################################")
        #pprint.pprint(data)
        #############
        # Add to db #
        #############
        try:
            session.add(TFTable(**data))
            session.commit()
            print(data["id"], " INSERTED!")
        except Exception as err:
            session.rollback()
            print("# NOT INSERTED: ", err)
        print("############################################################")


    #####################
    #  Tweet 2 Vec (?)  #
    #####################
    #pprint.pprint(data)
    print("############################################################")
