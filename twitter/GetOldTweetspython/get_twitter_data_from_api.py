import sys
sys.path.append("..")
sys.path.append("../..")
import twitter_config
import pprint
import postgis_functions
import pois_storing_functions
import tweepy
import json
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


# def get_fsq_points(c):


def get_tweets_from_loc_since_until_radius(l, since, until, rad):
    tweetCriteria = got.manager.TweetCriteria()\
        .setNear(l)\
        .setWithin(rad)\
        .setSince(since)\
        .setUntil(until)\
        .setMaxTweets(0)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    return tweets
# bebop google 52.010858,4.359553


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
    res["lang"] = tweet.lang
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


if __name__ == '__main__':
    api = setup_api()
    city = "ams"
    source = "twitter"
    count=0
    logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_places_point_iterations.txt"
    #not_inserted_file = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_whole_clipped_not_inserted.txt"
    while count<1:
        try:
            last_searched_id = pois_storing_functions.get_last_id_from_logfile(logfile)
            #print(last_searched_id)
            points = postgis_functions.get_pois_from_db("gt_fsq_ams_matched", last_searched_id)
            session, TTable = pois_storing_functions.setup_db("twitter_ams_places",
                                                                "twitter_ams_places_count", "twitter")
            rad = "0.05km"
            since = "2016-01-01"
            until = "2018-12-19"
            for fsq in points:
                print(fsq["rn"])
                log_last_searched_point(logfile, fsq["rn"])
                print("POINT: ", fsq["rn"], fsq["id"], fsq["lat"], fsq["lng"])
                ll = str(fsq["lat"]) + "," + str(fsq["lng"])
                tweets = get_tweets_from_loc_since_until_radius(ll, since, until, rad)
                print("ok")
                for tweet in tweets:
                    # print(dir(tweet))
                    # print(tweet.geo)
                    # print(tweet.hashtags)
                    # print(tweet.date)
                    tweet_by_id = api.get_status(tweet.id)
                    insert_tweet_to_db(fsq["rn"], fsq["id"], api.get_status(tweet.id), session, TTable)
        except Exception as err:
            count+=1
            print(count)
            print(err)


