import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pickle
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import postgis_functions
# Special thanks to
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#17howtofindtheoptimalnumberoftopicsforlda


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def get_processed_lda_reviews_from_db(table, source, lang, load):
    if load:
        return load_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/" +
                                lang + "_text_" + source )
    else:
        unproc_text = postgis_functions.get_lda_text_per_lang(table, lang)
        text = [t["processedtextlda"].split() for t in unproc_text]
        # Store reviews
        save_with_pickle("/home/bill/Desktop/thesis/code/UDS/textual_features/objects/" +
                         lang + "_text_" + source, text)
        return text


def load_with_pickle(filename):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def save_with_pickle(filename, obj):
    pickle_name = filename + ".pkl"
    with open(pickle_name, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    path = "/home/bill/Desktop/thesis/code/UDS/textual_features/objects/"
    # select tweets or reviews
    source_list = ["tweets", "rev"]
    # select language {en, nl}
    lang_list = ["en"]
    city = "ath"
    # select table
    for source in source_list:
        if source == "rev":
            table = "matched_places_google_reviews_" + city
        else:
            table = "matched_places_twitter_" +  city
        for lang in lang_list:
            print("Working... Source: " + source + ", Lang: " + lang)
            dict = corpora.Dictionary.load(path + lang + "_dict_" + source + ".pkl")
            term_matrix = corpora.MmCorpus(path + lang + "_term_matrix_" + source + ".mm")
            # text = get_processed_lda_reviews_from_db(table, source, lang, load=False)
            text = postgis_functions.get_lda_text_per_lang(table,lang)
            text = [t["processedtextlda"] for t in text if t["processedtextlda"] is not None]

            print(text)
            # text = [t["processedtextlda"].split() for t in unproc_text]

            mallet_path = '/home/bill/Desktop/thesis/packages/mallet-2.0.8/bin/mallet'
            # update this path
            #ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
            limit = 40
            start = 2
            step = 2
            # Can take a long time to run.
            print(dict)
            print(text)
            print(term_matrix)
            model_list, coherence_values = compute_coherence_values(dictionary=dict, corpus=term_matrix,
                                                                    texts=text,
                                                                    start=start, limit=limit, step=step)
            save_with_pickle("model_list_" + source + "_" + city, model_list)
            save_with_pickle("coherence_" + source + "_" + city, coherence_values)

            x = range(start, limit, step)
            # Print the coherence scores
            for m, cv in zip(x, coherence_values):
                print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
            # Show graph

            plt.plot(x, coherence_values)
            if source == "rev":
                title_source = "Reviews"
            else:
                title_source = "Tweets"
            if lang =="en":
                 title_lang = "English"
            else:
                title_lang = "Dutch"
            plt.title(title_lang + " " + title_source + " LDA Model")
            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            plt.savefig("/home/bill/Desktop/" +
                        lang + "_" + source + "_lda_" + str(step) + "_" + str(limit))

