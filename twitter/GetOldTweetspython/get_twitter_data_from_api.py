import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../textual_features/")
#print(sys.path)
import twitter.twitter_config as twitter_config
import pprint
import postgis_functions
from pois_storing_functions import setup_db, get_last_id_from_logfile

import tweepy
import json
import psycopg2
#import textual_features.textual_features_extraction.get_text_language as get_text_language
import langid

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got


def setup_api():
    ckey = twitter_config.ckey
    csecret = twitter_config.csecret
    atoken = twitter_config.atoken
    asecret = twitter_config.atoken
    OAUTH_KEYS = {'consumer_key': ckey, 'consumer_secret': csecret,
                  'access_token_key': atoken, 'access_token_secret': asecret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])

    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def get_tweets_from_loc_since_until_radius(l, since, until, rad):
    tweetCriteria = got.manager.TweetCriteria()\
        .setNear(l)\
        .setWithin(rad)\
        .setSince(since)\
        .setUntil(until)\
        .setMaxTweets(0)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    #, proxy='74.209.243.116:3128'
    return tweets
# bebop google 52.010858,4.359553


def get_text_language(text):
    # get text's language
    lang = langid.classify(text)[0]
    return lang


def insert_tweet_to_db(point_id, places_id, tweet, session, TTable):
    res = {}
    res["id"] = str(tweet.id) + "_" + places_id
    res["pointid"] = point_id
    res["fsqid"] = places_id
    res["createdat"] = tweet.created_at
    res["year"] = tweet.created_at.year
    res["month"] = tweet.created_at.month
    res["day"] = tweet.created_at.day
    res["hour"] = tweet.created_at.hour
    res["lang"] = get_text_language(tweet.text)
    res["text"] = tweet.text
    res["favoritecount"] = tweet.favorite_count
    res["retweetcount"] = tweet.retweet_count
    if tweet.coordinates:
        res["lat"] = tweet.coordinates["coordinates"][1]
        res["lng"] = tweet.coordinates["coordinates"][0]
        res["geom"] = 'POINT({} {})'.format(res["lng"], res["lat"])
    else:
        res["lat"], res["lng"], res["geom"] = None, None, None
    res["json"] = json.dumps(tweet._json)
    try:
        session.add(TTable(**res))
        session.commit()
        print("~~ ", res["pointid"], res["id"], " INSERTED!")

    except Exception as err:
        session.rollback()
        print("# NOT INSERTED: ", err)


def log_last_searched_point(logfile, originalpointindex):
    with open(logfile, "w") as text_file:
        text_file.write("Last searched point \n" + str(originalpointindex))
        text_file.close()

# def get_fsq_points_with_no_tweets(fsqtab, tweetab):
#     conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
#     c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#     c.execute("SELECT  * from {fsqtab} "
#               "WHERE "
#               "id  not in "
#               "(SELECT fsqid from {tab_twitter} "
#               " "
#               "OR "
#               "id NOT IN (SELECT fsqid from {tab_twitter})"
#               "ORDER BY RANDOM()".format(n=num, fsqtab=fsqtab, tab_twitter=tab_twitter))


def get_fsq_points_less_tweets_than(num, tab, tab_twitter, random):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if random:
        c.execute("SELECT  * from {tab} "
                  "WHERE "
                  "id  in "
                  "(SELECT fsqid from {tab_twitter} "
                  "GROUP BY fsqid "
                  "having COUNT(*)<{n}) "
                  "OR "
                  "id NOT IN (SELECT DISTINCT fsqid from {tab_twitter} WHERE year='2017' or year='2018')"
                  "ORDER BY RANDOM()".format(n=num, tab=tab, tab_twitter=tab_twitter))
    else:
        c.execute("SELECT  * from {tab} "
                  "WHERE "
                  "id  in "
                  "(SELECT fsqid from {tab_twitter} "
                  "GROUP BY fsqid "
                  "having COUNT(*)<{n}) "
                  "OR "
                  "id NOT IN (SELECT DISTINCT fsqid from {tab_twitter})"
                  "ORDER BY originalpointindex".format(n=num, tab=tab, tab_twitter=tab_twitter))

    return c.fetchall()


if __name__ == '__main__':
    api = setup_api()
    city = "ams"
    source = "twitter"
    count=0
    logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_matched.txt"
    while count<1:
        try:
            last_searched_id = get_last_id_from_logfile(logfile)
            #print(last_searched_id)
            #points = postgis_functions.get_pois_from_fsq_db("matched_fsq_" + city, last_searched_id)
            session, TTable = setup_db("matched_places_twitter_" + city,
                                        "twitter_" + city + "_places_count", "twitter")
            points = get_fsq_points_less_tweets_than(1, "matched_places_fsq_" + city, "matched_places_twitter_" + city, random=True)
            rad = "0.05km"
            since = "2017-01-01"
            until = "2018-12-19"
            # since = "2014-01-01"
            # until = "2016-12-30"
            count=0
            print(len(points))
            for fsq in points:
                count+=1
                print("COUNT: ", count)
                log_last_searched_point(logfile, fsq["point"])
                print("POINT: ", fsq["point"], fsq["id"], fsq["lat"], fsq["lng"])
                ll = str(fsq["lat"]) + "," + str(fsq["lng"])
                tweets = get_tweets_from_loc_since_until_radius(ll, since, until, rad)
                print(tweets)
                print("Got tweets")
                # if len(tweets)>100:
                #     tweets = tweets[0:100]
                print(len(tweets))
                for tweet in tweets:
                    tweet_by_id = api.get_status(tweet.id)
                    insert_tweet_to_db(fsq["point"], fsq["id"], api.get_status(tweet.id), session, TTable)
        except Exception as err:
            count+=1
            print(count)
            print(err)


