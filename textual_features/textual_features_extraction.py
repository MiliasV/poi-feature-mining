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


def get_sent_from_polyglot(eng_tweets, nl_tweets):
    t_en = [Text(t) for t in eng_tweets]
    t_en = change_language_polyglot_list(t_en, "en")
    t_nl = [Text(t) for t in nl_tweets]
    t_nl = change_language_polyglot_list(t_nl, "nl")

    eng_sent = np.mean([t.polarity for t in t_en])
    #except:
    #    eng_sent = None
    try:
        nl_sent = np.mean([t.polarity for t in t_nl])
    except:
        nl_sent = None
    return eng_sent, nl_sent


def get_sent_from_textblob(eng_tweets):
    tb = [TextBlob(emoji.demojize(t)) for t in eng_tweets]
    return np.mean([t.polarity for t in tb]) , np.mean([t.subjectivity for t in tb])


def get_text_trans_in_eng(t, text_list):
    # convert emojis to string and then translate
    return [t.translate(emoji.demojize(tweet)).text for tweet in text_list]


def train_lda_models(eng_tweets, nl_tweets, ntopics, passes):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    eng_dict = corpora.Dictionary(eng_tweets)
    nl_dict = corpora.Dictionary(nl_tweets)
    # Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    eng_term_matrix = [eng_dict.doc2bow(doc) for doc in eng_tweets]
    nl_term_matrix = [nl_dict.doc2bow(doc) for doc in nl_tweets]
    ldamodel_eng = Lda(eng_term_matrix, num_topics=ntopics, id2word=eng_dict, passes=passes)
    ldamodel_nl = Lda(nl_term_matrix, num_topics=ntopics, id2word=nl_dict, passes=passes)
    return ldamodel_eng, ldamodel_nl, eng_dict, nl_dict


def get_lda_models(eng_tweets, nl_tweets, ntopics, passes, load):
    if load:
        eng_dict = corpora.Dictionary.load("eng_dict.pkl")
        nl_dict = corpora.Dictionary.load("nl_dict.pkl")
        lda_eng = models.LdaModel.load("lda_eng.model")
        lda_nl = models.LdaModel.load("lda_nl.model")
    else:
        # Train the LDA model
        lda_eng, lda_nl, eng_dict, nl_dict = train_lda_models(eng_tweets, nl_tweets, ntopics, passes)
        lda_eng.save("lda_eng.model")
        lda_nl.save("lda_nl.model")
        eng_dict.save("eng_dict.pkl")
        nl_dict.save("nl_dict.pkl")
    return lda_eng, lda_nl, eng_dict, nl_dict


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


def get_text_tweets(table,  eng_stop, nl_stop, eng_exclude, nl_exclude, lemma):
    eng_tweets = []
    nl_tweets = []
    count=0
    tweets = postgis_functions.get_rows_from_table(table)
    for t in tweets:
        count+=1
        print(count)
        lang = get_text_language(t["text"])
        if lang == "en":
            # remove stopwords, punctuation, links and lemmatize words
            eng_tweets.append(clean(t["text"], eng_stop, eng_exclude, lemma).split())
        elif lang == "nl":
            # remove stopwords, punctuation, links and lemmatize words
            nl_tweets.append(clean(t["text"], nl_stop, nl_exclude, lemma).split())
    return eng_tweets, nl_tweets


def add_processed_lda_text_tweets(table):
    # Process tweets for LDA-based topic modelling - Load = True uses saved data
    # stop words and punctuations
    lemma, eng_stop, eng_exclude = setup_for_topic_modeling("en")
    lemma, nl_stop, nl_exclude = setup_for_topic_modeling("nl")
    count=0
    tweets = postgis_functions.get_rows_from_table_where_col_is_null(table, "processedtextlda")
    print(len(tweets))
    for t in tweets:
        lang = t["lang"]
        if lang == "en":
            print(t)
            count += 1
            print(count)
            # remove stopwords, punctuation, links and lemmatize words
            #eng_tweets.append(clean(t, eng_stop, eng_exclude, lemma).split())
            postgis_functions.add_processed_text_to_table(clean(t["text"], eng_stop, eng_exclude, lemma),
                                                          "processedtextlda", table, t["id"])
        elif lang == "nl":
            print(t)
            count += 1
            print(count)
            # remove stopwords, punctuation, links and lemmatize words
            #nl_tweets.append(clean(t, nl_stop, nl_exclude, lemma).split())
            postgis_functions.add_processed_text_to_table(clean(t["text"], nl_stop, nl_exclude, lemma),
                                                          "processedtextlda", table, t["id"])


def get_processed_lda_tweets_from_db(table, load):
    if load:
        eng_tweets_text = load_with_pickle("eng_tweets_text")
        nl_tweets_text = load_with_pickle("nl_tweets_text")
        return eng_tweets_text, nl_tweets_text
    else:
        eng_tweets = postgis_functions.get_tweets_lda_text_per_lang(table, "en")
        eng_tweets_text = [t["processedtextlda"].split() for t in eng_tweets]
        nl_tweets = postgis_functions.get_tweets_lda_text_per_lang(table, "nl")
        nl_tweets_text = [t["processedtextlda"].split() for t in nl_tweets]
        # Store tweets
        save_with_pickle("eng_tweets_text", eng_tweets_text)
        save_with_pickle("nl_tweets_text", nl_tweets_text)
        return eng_tweets_text, nl_tweets_text


def get_processed_tweets(table, load):
    if load:
        eng_tweets = load_with_pickle("eng_tweets")
        nl_tweets = load_with_pickle("nl_tweets")
        return eng_tweets, nl_tweets
    else:
        # stop words and punctuations
        lemma, eng_stop, eng_exclude = setup_for_topic_modeling("en")
        lemma, nl_stop, nl_exclude = setup_for_topic_modeling("nl")
        # takes around 5 mins (saved for now)
        eng_tweets, nl_tweets = get_text_tweets(table, eng_stop, nl_stop,
                                                eng_exclude, nl_exclude, lemma)
        # Store tweets
        save_with_pickle("eng_tweets", eng_tweets)
        save_with_pickle("nl_tweets", nl_tweets)
        return eng_tweets, nl_tweets


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
        count += 1
        print(count)
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
        tweets_text = [t["processedtextlda"].split() for t in tweets]
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
        #print(place_topics)
    return place_topics


if __name__ == '__main__':
    count = 0
    format_str = "%Y-%m-%d %H:%M:%S"
    t = Translator()
    # get places
    fpoints = postgis_functions.get_rows_from_table("matched_fsq_ams")

    # add processed tweets for lda to db
    # add_processed_lda_text_tweets("matched_twitter_ams")
    # get processed tweets for lda per language
    print("LOADING Tweets....")
    num_topics = 5
    eng_tweets, nl_tweets = get_processed_lda_tweets_from_db("matched_twitter_ams", load=True)
    print("TRAINING Model...")
    # train or load models
    lda_eng, lda_nl, eng_dict, nl_dict = get_lda_models(eng_tweets, nl_tweets,
                                                        ntopics=num_topics, passes=20, load=True)
    # print(lda_eng.show_topic(topic[0], topn=5))
    # for every matched place
    for f in fpoints:
        data = {}
        fpoint = {k: v if v is not None else "" for k, v in f.items()}
        data["type"] = postgis_functions.get_type_of_place(fpoint)
        print(fpoint)
        data = get_time_diffs(data, "matched_twitter_ams", fpoint["id"])

        ######################
        # LDA Topic Modeling #
        ######################
        #  get tweets per place per lang
        eng_tweets_lda = postgis_functions.get_tweets_lda_text_per_lang_from_fsqid("matched_twitter_ams", fpoint["id"], "en")
        nl_tweets_lda = postgis_functions.get_tweets_lda_text_per_lang_from_fsqid("matched_twitter_ams", fpoint["id"], "nl")
        print("GETTING Topics from ", len(eng_tweets_lda) + len(nl_tweets_lda), " Tweets")
        # print(lda_eng.show_topics(num_topics=5))
        eng_topics = get_topics_from_lda(eng_tweets_lda, lda_eng, eng_dict, num_topics)
        nl_topics = get_topics_from_lda(nl_tweets_lda, lda_nl, nl_dict, num_topics)
        # for i in range(len(eng_topics)):
        #     print(eng_topics[i], lda_eng.show_topic(i, topn=5))
        #print(lda_nl.show_topics())
        #print(nl_topics)
        for i, val in enumerate(eng_topics):
            data["topiceng" + str(i+1)] = eng_topics[i]
            data["topicnl" + str(i+1)] = nl_topics[i]
            data["topic" + str(i+1)] = (eng_topics[i] + nl_topics[i]) /2.0

        ####################
        # Tweet statistics #
        ####################
        # get unprocessed text
        eng_tweets = postgis_functions.get_tweets_text_per_lang("matched_twitter_ams", fpoint["id"], "en")
        nl_tweets = postgis_functions.get_tweets_text_per_lang("matched_twitter_ams", fpoint["id"], "nl")
        # count of tweets - Think about it as I gather until 50-100 tweets per place
        data["entweetcount"] = len(eng_tweets)
        data["nltweetcount"] = len(nl_tweets)
        data["totaltweetcount"] = data["entweetcount"] + data["nltweetcount"]

        print(eng_tweets)
        print(nl_tweets)
        eng_tweets = [x["text"] for x in eng_tweets]
        nl_tweets = [x["text"] for x in nl_tweets]
        print(eng_tweets)
        print(nl_tweets)

        # nl_tweets_flat = [x for y in nl_tweets for x in y]

        # count of words
        data["enwordcount"] = sum([len(x.split(" ")) for x in eng_tweets])
        data["nlwordcount"] = sum([len(x.split(" ")) for x in nl_tweets])
        data["totalwordcount"] = data["enwordcount"] + data["nlwordcount"]

        # avg. count of words
        data["engavgword"] = data["enwordcount"] / data["entweetcount"]
        data["nlavgword"] = data["nlwordcount"] / data["nltweetcount"]
        data["avgword"] = (data["engavgword"] + data["nlavgword"]) / 2.0
        print(eng_tweets)

        tweetimes = postgis_functions.get_daytime_by_id("matched_twitter_ams", fpoint["id"])
        pprint.pprint(data)
        print("############################################################")

    ######################
    # Sentiment analysis #
    ######################
    # POLYGLOT
    # print(eng_tweets)
    # print(nl_tweets)
    # eng_sent, nl_sent = get_sent_from_polyglot(eng_tweets, nl_tweets)
    # print(eng_sent)
    # print(nl_sent)

    ############
    # TextBlob #
    ############
    # for english tweets
    data["enpol"], data["ensubj"] = get_sent_from_textblob(eng_tweets)
    # for dutch tweets
    # translate
    if nl_tweets:
        nl_trans_tweets = get_text_trans_in_eng(t, nl_tweets)
        data["nlpol"], data["nlsub"] = get_sent_from_textblob(nl_trans_tweets)
    else:
        data["npol"], data["nlsub"] = None, None


    ##################
    # Topic Modeling #
    ##################
    data["entopics"], nltopicsdutch = get_topics_from_lda(eng_tweets, nl_tweets, ntopics=2, nwords=1)
    # translate topics
    data["nltopics"] = [t.translate(str(word), src="nl", dest="en").text for word in nltopicsdutch]
    #print(data)
    #data["processedtext"] = tweet2vec.preprocess(t["text"].rstrip())

    ##################
    #  Tweet 2 Vec   #
    ##################
    pprint.pprint(data)
    print("############################################################")
