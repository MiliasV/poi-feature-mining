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


if __name__ == '__main__':
    # Store reviews to table and process text for lda
    store_google_reviews_and_processed_text()
    gpoints = postgis_functions.get_rows_from_table("matched_google_ams")
    # for g in gpoints:
    #     rpoints = postgis_functions.get_row_by_id("matched_google_reviews_ams")