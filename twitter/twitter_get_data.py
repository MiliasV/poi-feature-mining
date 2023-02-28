from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import urllib.request
import tweepy
import pprint
import numpy as np
import cv2
import os


def show_save_img(img, show_flag, save_flag, duration):
    if show_flag:
        filename = "Amsterdam_Twitter"
        # Load an color image in grayscale

        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)  # Create a named window
        cv2.moveWindow(filename, 100, 500)  # Move it to (40,30)
        #img = cv2.imread(filename)
        cv2.imshow(filename, img)
        cv2.waitKey(duration)
        cv2.destroyAllWindows()
    #if not save_flag:
        #os.remove(filename)

# OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
ckey="QsYxUreMW2d6pOX5SO2GFFJVb"
csecret="ll9qmhV7Tq2qd0eZLsDDHC15tVt0fF7b64ldenT2fXdYuuEZNx"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
atoken="782617496884879360-b3NJx6afLpJlIUROtPpNpaL7XGeLSbs"
asecret="cIbgbWO1HXnFvQhMt0rVBcZSZSEVwqOLNU9MisQIoo7uJ"

OAUTH_KEYS = {'consumer_key':ckey, 'consumer_secret':csecret,
 'access_token_key':atoken, 'access_token_secret':asecret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])

api = tweepy.API(auth, wait_on_rate_limit=True)

IMAGE_TYPES = ["jpg", "png", "webp", "gif"]
#curs_tweet = tweepy.Cursor(api.search, geocode="52.38634009502191,4.873498724566273,0.1km").items()#, include_entities=True).items()
curs_tweet = tweepy.Cursor(api.search, "999584980090150912").items()#, include_entities=True).items()
#cricTweet = tweepy.Cursor(api.search, q='Amsterdam', geocode="52.370216,4.895168,5km", include_entities=True).items(1000)

pp = pprint.PrettyPrinter(indent=4)

for tweet in curs_tweet:
    print(tweet.text)
    print(tweet.created_at)
    print(tweet.geo)
    if ('media' in tweet.entities):# and (tweet._json["place"]):
        #pp.pprint(tweet._json['place'])
        #print(tweet.created_at)
        #print(tweet.text)
        #print(tweet.lang)
        #print("\n")
        for image in tweet.entities['media']:
            img = url_to_image(image['media_url'])
            #show_save_img(img, True, False, 2600)

            #     if "jpg" in image:
        #     #if any(x in image for x in IMAGE_TYPES):
        #         print(image['media_url'])
        #print(tweet.entities['media'])
        #print(image['media_url'])

