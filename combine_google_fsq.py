import sys
sys.path.append("..")
import postgis_functions

from difflib import SequenceMatcher
from pyxdameraulevenshtein import  normalized_damerau_levenshtein_distance
import fuzzy
import Levenshtein
from nltk.corpus import wordnet
from transliterate import translit


def longest_substring(str1, str2):
    # source: https://www.geeksforgeeks.org/sequencematcher-in-python-for-longest-common-substring/
    # initialize SequenceMatcher object with
    # input string
    seq_match = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seq_match.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    if (match.size != 0):
        return str1[match.a: match.a + match.size]
    else:
        return ""


def get_ro_similarity(a, b):
    # based on Ratcliff and Obershelp's algorithm
    return SequenceMatcher(None, a, b).ratio()


def get_name_in_latin_alph(name):
    try:
        return translit(name, reversed=True)
    except:
        return name


def get_levenshtein_phonetic_similarity(osm_name, source_name):
    print(osm_name, source_name)
    dmeta = fuzzy.DMetaphone()
    try:
        dmeta_osm = dmeta(osm_name)[0]#.decode("utf-8")
        dmeta_source = dmeta(source_name)[0]#.decode("utf-8")
    except:
        return None
    return Levenshtein.ratio(dmeta_osm, dmeta_source)


def get_str_similarity(source, target):
    similarities = {
        "long_substring": longest_substring(source, target),
        "ro_similarity": get_ro_similarity(source, target),
        "dleven_similarity": 1 - normalized_damerau_levenshtein_distance(source, target),
        "leven_similarity": Levenshtein.ratio(source, target),
        "phonetic_similarity": get_levenshtein_phonetic_similarity(source, target),
    }
    similarities["len_long_substring"] = len(similarities["long_substring"])
    return similarities


def get_gf_similarity_score(google_info, fsq_info):
    # calculate the google-foursquare similarity score
    # (and not the girlfriend similarity score! :p )
    print("ok")

if __name__ == "__main__":

    cg = postgis_functions.connect_to_db()
    cf = postgis_functions.connect_to_db()

    # get long/lat from google pois
    #gpoints = postgis_functions.get_row_and_ll_of_pois(cg, "google_ams_center_40")
    fpoints = postgis_functions.get_row_and_ll_of_pois(cf, "fsq_ams_center_40")

    # choose radius
    rad = 20
    ###############
    # FOR EACH POI#
    ###############
    for i, lat, lng in fpoints:
        #fsq_closest_points = postgis_functions.get_matching_attr_from_pois_within_radius(cf, "fsq_ams_center_40",
         #                                                                (lat, lng), rad)
        google_closest_points = postgis_functions.get_matching_attr_from_pois_within_radius\
            (cg, "google_ams_center_40", (lat, lng), rad)
        #google_row = postgis_functions.get_row_from_index(cg, "google_ams_center_40", i)
        fsq_row = postgis_functions.get_row_from_index(cf, "fsq_ams_center_40", i)

        #print("~ GOOGLE: ", google_row[0][1:5], google_row[0][8], google_row[0][10])
        print("~ FSQ: ", fsq_row[0][1:5], fsq_row[0][8:10])

        for google_row in google_closest_points:
            print("~ GOOGLE :", google_row[1:5], google_row[8:10])
        print("#################################################################################")

            # for row in fsq_closest_points_c:
        #     print(row)

