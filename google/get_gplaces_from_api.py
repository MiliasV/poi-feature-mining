import sys
sys.path.append("..")
import google_config
#import google.google_config as google_config
from googleplaces import GooglePlaces, types, lang
import sqlite3
import pprint
import simplejson as json
import populartimes
import pois_storing_functions
import get_map_points_to_search
import urllib

##########################################################
# Script for finding and storing POIs from Google Places #
##########################################################


def setup():
    API_KEY = google_config.api_key
    google_places = GooglePlaces(API_KEY)
    return  google_places, API_KEY


def extract_data_from_json(google_json):
    res = {}
    res["name"] = google_json["name"]
    res["id"] = google_json["place_id"]
    #res["pop_times"] = google_json["poptimes"]
    res["rating"] = None
    if "rating" in google_json:
        res["rating"] = google_json["rating"]
    res["website"] = None
    if "website" in google_json:
        res["website"] = google_json["website"]
    res["streetlng"], res["streetnumlng"], res["streetsrt"], res["streetnumsrt"] = None, None, None, None
    for addr_details in google_json["address_components"]:
        if addr_details["types"][0] == "route":
            res["streetlng"] = addr_details["long_name"]
            res["streetsrt"] = addr_details["short_name"]
        elif addr_details["types"][0] == "street_number":
            res["streetnumlng"] = addr_details["long_name"]
            res["streetnumsrt"] = addr_details["short_name"]
    res["rscount"] = 0
    res["phcount"] = 0
    res["poptimes"] = google_json["poptimes"]
    res["phone"] = None
    if "international_phone_number" in google_json:
        res["phone"] = google_json["international_phone_number"]
    if "reviews" in google_json:
        res["rscount"] = len(google_json["reviews"])
    if "photos" in google_json:
        res["phcount"] = len(google_json["photos"])
    # at least 5 types to be inserted
    while len(google_json["types"])<5:
        google_json["types"].append(None)
    for i in range(5):
        res["type" + str(i+1)] = google_json["types"][i]
    res["lat"] = google_json["geometry"]["location"]["lat"]
    res["lng"] = google_json["geometry"]["location"]["lng"]
    return res


def insert_data(session, GTable, search_lat, search_lng, google_json, ogc_fid,
                count_places, count_duplicates):
    res = extract_data_from_json(google_json)
    res["json"] = json.dumps(google_json)
    res["geom"] = 'POINT({} {})'.format(res["lng"], res["lat"])
    res["searchlat"] = search_lat
    res["searchlng"] = search_lng
    res["originalpointindex"] = ogc_fid
    try:
        session.add(GTable(**res))
        session.commit()
        count_places += 1
        print("~~ ", google_json["name"], " INSERTED!")

    except Exception as err:
        session.rollback()
        count_duplicates += 1
        print("# NOT INSERTED: ", err)
    return count_places, count_duplicates


def insert_count_data(session, CTable, poi_number, count_places, count_dupl):
    try:
        session.add(CTable(id=poi_number, countplaces=count_places, countduplicates=count_dupl))
        session.commit()
    except Exception as err:
        session.rollback()
        print("# [COUNT] NOT INSERTED: ", err)


def get_places_by_ll(google_places, ll, rad):
    return google_places.nearby_search(lat_lng=ll, radius=rad)


def get_pop_times_from_id(api_key, place_id):
    try:
        pop_times = populartimes.get_id(api_key, place_id)
    except:
        return None
    if "populartimes" in pop_times:
        return json.dumps(pop_times["populartimes"])
    else:
        return None


def add_pop_times_in_places(api_key, query_results):
    # for each place gotten from google
    for place in query_results.places:
        place.get_details()
        gid = place.details["place_id"]
        place.details["poptimes"] = get_pop_times_from_id(api_key, gid)
    return query_results.places


if __name__ == "__main__":
    google_places, api_key = setup()
    c, rad, logfile, errorfile = get_map_points_to_search.config_parameters_for_searching("google", "ath")
    last_searched_id = pois_storing_functions.get_last_id_from_logfile(logfile)
    # define which table
    session, GTable, CTable = pois_storing_functions.setup_db("google_ath_whole_clipped_40",
                                                         "google_ath_whole_clipped_count", "google")
    #define types we don't care about
    google_not_wanted_types = ["route"]
    # For each point --> search nearby in google
    for ogc_fid, point_lat, point_lng in c:
    #try:
        count_places = 0
        count_duplicates = 0
        print("POINT: ", ogc_fid, point_lat, point_lng)
        # keep the id of the last searched point
        get_map_points_to_search.log_last_searched_point(logfile, ogc_fid)
        ll = {"lat": str(point_lat), "lng": str(point_lng)}
        query_results = google_places.nearby_search(lat_lng=ll, radius=rad)
        # if I want to include popular times
        places_extended = add_pop_times_in_places(api_key, query_results)
        # for each place gotten from google
        for place in places_extended:
                # put a try except
            place.get_details()
            google_json = place.details
            if not any(x in google_not_wanted_types for x in google_json["types"]):
                count_places, count_duplicates = insert_data(session, GTable, point_lat,
                                                             point_lng, google_json, ogc_fid,
                                                             count_places, count_duplicates)
        insert_count_data(session, CTable, ogc_fid, count_places, count_duplicates)
        # except Exception as err:
        #     with open(errorfile, "a+") as text_file:
        #         print(f"ERROR \n{err}", file=text_file)
        #         text_file.close()
