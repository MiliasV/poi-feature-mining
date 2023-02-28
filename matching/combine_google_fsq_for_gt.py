import sys
sys.path.append("..")
import postgis_functions
from difflib import SequenceMatcher
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import fuzzy
import Levenshtein
from nltk.corpus import wordnet
from transliterate import translit
import pois_storing_functions
from urllib.parse import urlparse
import combine_google_fsq_with_model
import create_similarities_table
import pandas as pd
from romanize import romanize
errcount = 0

# Match Google & Foursquare POIs for creating ground truth
# (Strict matching rules are used)


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
    dmeta = fuzzy.DMetaphone()
    try:
        dmeta_osm = dmeta(osm_name)[0]#.decode("utf-8")
        dmeta_source = dmeta(source_name)[0]#.decode("utf-8")
        return Levenshtein.ratio(dmeta_osm, dmeta_source)

    except Exception as err:
        return None


def get_str_similarity(source, target):
    source = source.lower()
    target = target.lower()
    long_sub = longest_substring(source, target)
    #"long_substring": longest_substring(source, target),
    similarities = {
        "ro_similarity": get_ro_similarity(source, target),
        "dleven_similarity": 1 - normalized_damerau_levenshtein_distance(source, target),
        "leven_similarity": Levenshtein.ratio(source, target),
        "phonetic_similarity": get_levenshtein_phonetic_similarity(source, target),
    }
    if (len(long_sub)):
        similarities["len_long_substring"] = len(long_sub)/len(source)
    else:
        similarities["len_long_substring"] = 0
    res = {k: v for k, v in similarities.items() if v is not None}
    avg_sim = sum(res.values())/len(res.values())
    return avg_sim


def get_gf_similarity_score(google_info, fsq_info):
    # calculate the google-foursquare similarity score
    # (and not the girlfriend similarity score! :p )
    print("ok")


def match_by_phone(fpoint, gpoint, thres):
    try:
        if get_str_similarity(fpoint["phone"], gpoint["phone"])>=thres:
            return True
    except:
        pass
    return False


def match_by_addr(fpoint, gpoint):
    try:
        if fpoint["street"] and fpoint["streetnum"] and \
                fpoint["street"] == gpoint["streetlng"] and fpoint["streetnum"] == gpoint["streetnumlng"]:
            return True
    except:
        pass
    return False


def match_by_name(fpoint, gpoint, thres):
    if get_str_similarity(fpoint["name"], gpoint["name"])>=thres:
        return True
    return False


def match_by_street(fpoint, gpoint, thres):
    try:
    #if "street" in fpoint and "streetlng" in gpoint and fpoint["street"] and gpoint["streetlng"]:
        if get_str_similarity(fpoint["street"], gpoint["streetlng"])>=thres:
            return True
    except:
        pass
    return False


def match_by_website(fpoint, gpoint, thres):
    try:
        fweb = urlparse(fpoint["website"])
        gweb = urlparse(gpoint["website"])
        fweb = fweb["netloc"]
        gweb = gweb["netloc"]
        if Levenshtein.ratio(fweb, gweb)>=thres:
            return True
    except:
        pass
    return False


def print_matched(fpoint, gpoint, attr):
    print("~~~~ MATCH ~~~~ ", attr)
    print("~FSQ: ", fpoint)
    print("~GOO: ", gpoint)


def add_matched_to_db(session, FTable, GTable, MTable, fpoint, gpoint, reason):
    match = {"id": fpoint["id"] + "_" + gpoint["id"], "fsqid": fpoint["id"],
             "googleid": gpoint["id"], "reason": reason, "point": fpoint["point"]}
    global errcount
    fpoint_for_type = {k: v if v is not None else "" for k, v in fpoint.items()}

    # type = postgis_functions.get_type_of_place(fpoint_for_type)
    # fpoint["type"] = type
    # gpoint["type"] = type
    try:
        session.add(FTable(**fpoint))
        session.add(GTable(**gpoint))
        session.add(MTable(**match))
        session.commit()
        print("~~ MATCH INSERTED!")
        # print(fpoint)
        # print(gpoint)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    except Exception as err:
        errcount+=1
        print("~~ MATCH NOT INSERTED", reason, fpoint["point"])
        print(fpoint)
        print(gpoint)
        print(err)
        session.rollback()


def convert_dict_to_latin(d, l):
    if l == "all":
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = romanize(v)
    else:
        for f in l:
            if d[f]:
                d[f] = romanize(d[f])
    return d


def model_predictions(clf, prev_predict_prob, fpoint, gpoint):
    sim_dict = create_similarities_table.get_similarities_dict(fpoint, gpoint)
    sim_df = pd.DataFrame.from_dict([sim_dict])
    sim_df = sim_df.fillna(-1)
    fmatched = {}
    gmatched = {}
    if clf.predict_proba(sim_df)[0][1] > 0.8:
        print(clf.predict_proba(sim_df))
        predict_prob = clf.predict_proba(sim_df)[0][1]
        # if previous match had larger probability don't change match
        if prev_predict_prob > predict_prob:
            print("~~~ Already matched!!!")
            print(fpoint)
            print(gpoint)
            # count_conflicts+=1
        else:
            print("~~ Matched")
            prev_predict_prob = predict_prob
            fmatched, gmatched = fpoint.copy(), gpoint.copy()
            print(fpoint)
            print(gpoint)
    return prev_predict_prob, fmatched, gmatched


def print_matched_count(countaddr, countphone, countnamestreet, countwebstreet, countwebname, countnamedist, count):
    print("#################################################################################")
    print("COUNT (addr): ", countaddr)
    print("COUNT (Phone): ", countphone)
    print("COUNT (Namestreet): ", countnamestreet)
    print("COUNT (webstreet): ", countwebstreet)
    print("COUNT (webname): ", countwebname)
    print("COUNT (namedist): ", countnamedist)
    #print("TOTAL: ", count)#countaddr +countphone +countnamestreet + countwebstreet + countwebname + countnamedist)
    print("COUNT (actual): ", count)
    print("@@@@@@@@@@@@@@@@@@@@@")


if __name__ == "__main__":
    city = "ath"
    # choose radius
    #for rad in [275]:

    fpoints = postgis_functions.get_pois_for_matching("fsq_" + city + "_whole__clipped_40", 0)
    #fpoints = postgis_functions.get_pois_for_matching("fsq_" + city + "_places", 0)

    # Create matched tables
    session, FTable = pois_storing_functions.setup_db("tmatched_places_fsq_" + city, "", "fsq_matched")
    session.close()
    session, GTable = pois_storing_functions.setup_db("tmatched_places_google_"+ city, "", "google_matched")
    session.close()
    session, MTable = pois_storing_functions.setup_db("tmatched_places_google_fsq_" + city, "", "matching_table")
    rad = 300
    ###############
    # FOR EACH POI#
    ###############
    countnamestreet=0
    countwebstreet=0
    countphone=0
    countaddr=0
    point = 0
    countwebname = 0
    countnamedist = 0
    matching_score = 0
    count = 0
    thres = 7
    count_conflicts=0
    #clf = combine_google_fsq_with_model.get_trained_model("similarities_ams_table")
    for fpoint in fpoints:
        print_matched_count(countaddr, countphone, countnamestreet, countwebstreet, countwebname, countnamedist, count)
        #prev_predict_prob = 0
        # if greek POIs, convert to latin alphabet
        if city == "ath":
            fpoint = convert_dict_to_latin(fpoint, "all")
        score = 0
        # if point>253:
        # #     print(fpoint["originalpointindex"])
        #      break
        print("POINT ", point)
        print("COUNT ", count)
        print("Conflicts ", count_conflicts)
        # if point>1248:
        #     break

        point+=1
        # Gather places within radius
        google_closest_points = postgis_functions.get_matching_attr_from_pois_within_radius\
             ("google_" + city + "_whole_clipped_40", fpoint["lat"], fpoint["lng"], rad)
        for gpoint in google_closest_points:
            # # if greek POIs, convert to latin alphabet
            if city == "ath":
                gpoint = convert_dict_to_latin(gpoint, ["name", "streetsrt", "streetlng"])
            # model_predictions(clf, prev_predict_prob, fpoint, gpoint)
            # if prev_predict_prob > 0.8:
            #     print("ADDED TO DB")
            #     count += 1
            #     fmatched["point"] = point
            #     gmatched["point"] = point
            #     add_matched_to_db(session, FTable, GTable, MTable, fmatched, gmatched, "model-0.9")
            # if match_by_website(fpoint, gpoint, 0.8) and match_by_street(fpoint, gpoint, 0.8):
            #     if score< 3:
            #         fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "web_street",  3
            #         countwebstreet+=1
                # Match by phone  (if it exists) (total match!)
            if match_by_phone(fpoint, gpoint, 1) and match_by_name(fpoint, gpoint, 0.5) and postgis_functions.get_distance(fpoint, gpoint)<=40:
                if score < 6:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "phone_name", 6
                    print(fmatched)
                    print(gmatched)
                    countphone += 1
            # match by addr total and name 0.3
            # Remove!!!
            # if match_by_addr(fpoint, gpoint) and match_by_name(fpoint, gpoint, 0.7):
            #     if score < 5:
            #         fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "addr_name",  7
            #         countaddr+=1
            # match by name and street
            if match_by_name(fpoint, gpoint, 0.7) and match_by_street(fpoint, gpoint, 0.8):
                if score < 7:
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "street_name",  7
                    countnamestreet+=1
            if match_by_name(fpoint, gpoint, 0.7) and postgis_functions.get_distance(fpoint, gpoint)<=50:
                if score < 8:
                    print(fpoint)
                    print(gpoint)
                    fmatched, gmatched, reason, score = fpoint.copy(), gpoint.copy(), "dist_name",  8
                    countnamedist+=1
        if score > 0:
            count+=1
            fmatched["point"] = point
            gmatched["point"] = point
            add_matched_to_db(session, FTable, GTable, MTable, fmatched, gmatched, reason)
    print("RAD = ", rad)
    print("Total Points Matched = ", count , " / ", point) # countnamestreet + countwebstreet + countphone + countaddr + countwebname + countnamedist)
    print("ERRORS: ", count_conflicts)
