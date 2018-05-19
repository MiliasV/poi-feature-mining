import sqlite3
import osm_pois.get_osm_data as get_osm_data
import foursquare_my.get_foursquare_data as fsq
from difflib import SequenceMatcher
import google.get_gplaces_from_db as get_gplaces_from_db
import google.get_gplaces_from_api as get_gplaces_from_api
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import fuzzy
import Levenshtein
from nltk.corpus import wordnet
from transliterate import translit, get_available_language_codes
from geopy.distance import vincenty


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


def get_name_similarity(osm_name, source_name):
    similarities = {
        "long_substring": longest_substring(osm_name, source_name),
        "ro_similarity": get_ro_similarity(osm_name, source_name),
        "dleven_similarity": 1 - normalized_damerau_levenshtein_distance(osm_name, source_name),
        "leven_similarity": Levenshtein.ratio(osm_name, source_name),
        "phonetic_similarity" : get_levenshtein_phonetic_similarity(osm_name, source_name),
    }
    similarities["len_long_substring"] = len(similarities["long_substring"])
    return similarities


def get_osm_similarity_score(osm_info, source_info):
    num_attr = 0
    score = 0
    # Pre-processing
    osm_info["name"] = str(osm_info["name"]).lower()
    source_info["name"] = str(source_info["name"]).lower()
    # Make the alphabet latin for the Greek letters
    osm_name_lat = get_name_in_latin_alph(osm_info["name"])
    source_name_lat = get_name_in_latin_alph(source_info["name"])

    # compute name similarity
    sims = get_name_similarity(osm_name_lat, source_name_lat)
    # compute distance
    sims["distance"] = vincenty((osm_info["ll"][0][1], osm_info["ll"][0][2]), (source_info["lat"], source_info["lng"])).m

    for attr in ["name", "street", "street_num", "website"]:#, "type"]:
        if osm_info.get(attr) and source_info.get(attr):
            # check both long and short type description from foursquare
            if attr == "type":
                score += max((get_ro_similarity(str(osm_info[attr]), str(source_info[attr][0])),
                               get_ro_similarity(str(osm_info[attr]), str(source_info[attr][1]))))
            else:
                score += get_ro_similarity(str(osm_info[attr]), str(source_info[attr]))
            num_attr += 1
    return float(score)/num_attr, num_attr, sims


def get_matches_from_google(osm_info, rad):
    google_places, gapi_key = get_gplaces_from_api.setup()
    lat_lng = {"lat": str(osm_info["ll"][0][1]), "lng":str(osm_info["ll"][0][2])}
    query_results = get_gplaces_from_api.get_places_by_ll(google_places, lat_lng, rad)
    gg_info = {}
    for place in query_results.places:
        id = place.place_id
        gg_info[id] = get_gplaces_from_api.get_data_for_matching(place)
        gg_info[id]["score"], gg_info[id]["num_attr"], gg_info[id]["sim_dict"] = get_osm_similarity_score(osm_info, gg_info[id])
    return gg_info


def get_matches_from_foursquare(osm_info, radius):
    # translate ll for fourquare API
    ll = str(osm_info["ll"][0][1]) + "," + str(osm_info["ll"][0][2])
    # define fs client
    fsq_client = fsq.setup()
    # first filter: in that radius
    fsq_venues = fsq.get_venues_by_ll(fsq_client, ll, radius)
    return fsq.get_dict_with_scored_venues(fsq_venues, get_osm_similarity_score, osm_info)
        #photos = client.venues.photos(VENUE_ID="4efe05970e01089c53e3764a", params={})


def get_max_score_from_dict(dict):
    max = -1
    id = '1'
    for i, v in dict.items():
        if float(v["score"]) > max:
            max = v["score"]
            id = i
    return id


def show_matches(source, osm_info, rad):
    if source =="FSQ":
        matches = get_matches_from_foursquare(osm_info, rad)
    elif source == "Goo":
        matches = get_matches_from_google(osm_info, rad)
    if matches:
        best = get_max_score_from_dict(matches)
        print("~", source, "(", matches[best]["score"], ") : ", matches[best])
    else:
        print("~", source, ": Not Found")


if __name__ == "__main__":
    # databases
    ams_db ='../../databases/ams_bb.db'
    ath_db = '../../databases/ath_bb.db'
    # select osm db
    DB = ath_db
    # connect to osm db
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # select which pois'types we investigate
    pois_ids = get_osm_data.get_ids_by_category(c, "'amenity'", "('cinema', 'nightclub', 'theatre', "
                                                              "'bar', 'cafe', 'fast_food', 'pub', 'restaurant'"
                                                              "'pharmacy', 'hospital', 'university','school'"
                                                              "'cemetery', 'police', 'church', 'archaeologogical_site')")
    # choose radius (how far from the osm ll)
    rad = 20
    count = 0
    ###############
    # FOR EACH POI#
    ###############
    for poi in pois_ids[40:50]:
        osm_info = {}
        poi_id = poi[0]
        print("##########  POI ( ", poi_id," ) | Loop: ", count, "####################################################################")
        count += 1
        # get osm info
        osm_info["name"] = get_osm_data.get_name_from_id(c, poi_id)
        # continue if there is at least a name (restriction)
        if osm_info["name"]:
            osm_info = get_osm_data.get_data_for_matching(osm_info, c, poi_id)
            print("~ OSM : ", osm_info)
            show_matches("Goo", osm_info, rad)
            show_matches("FSQ", osm_info, rad)
            print("########################################################################################")

    # Foursquare Enrichment
    #osm_name[0][2], osm_type, osm_addr,