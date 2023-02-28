import postgis_functions
import pprint
import ast
import json
import pois_storing_functions
import google_fsq_features_extraction

if __name__ == '__main__':
    # initializations
    count = 0
    city = "ams"
    #format_str = "%Y-%m-%d %H:%M:%S"
    #session, GFTable = pois_storing_functions.setup_db("matched_gf_features_ams", "notused", "gf_features")
    gtab = "matched_places_google_" + city
    ftab = "matched_places_fsq_" + city
    # get places
    gfpoints = postgis_functions.get_google_fsq_features(gtab, ftab)
    session, GFTable = pois_storing_functions.setup_db("matched_places_spatial_features_" + city + "2", "notused", "spatial")

    #fpoints = postgis_functions.get_rows_from_id_not_in_table("matched_fsq_ams", "matched_text_features_ams", "id")

    # selecting what to store in table from the general information from google and fsq
    for gf in gfpoints:
        gfdata = {}
        gf["id"] = gf["fid"]
        # for using the function id is fsq id
        types_100 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 100)
        types_1000 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 1000)
        types_2000 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 2000)
        types_3000 = postgis_functions.get_place_types_in_radius(gf, "matched_places_fsq_" + city, 3000)

        gfdata = google_fsq_features_extraction.get_nearby_places(gfdata, types_100, "100")
        gfdata = google_fsq_features_extraction.get_nearby_places(gfdata, types_1000, "1000")
        gfdata = google_fsq_features_extraction.get_nearby_places(gfdata, types_2000, "2000")
        gfdata = google_fsq_features_extraction.get_nearby_places(gfdata, types_3000, "3000")
        gfdata["name"] = gf["gname"]
        gfdata["type"] = gf["type"]
        gfdata["point"] = gf["point"]
        gfdata["placesid"] = gf["gid"]
        gfdata["lat"] = gf["lat"]
        gfdata["lng"] = gf["lng"]

        print("############################################################")
        try:
            session.add(GFTable(**gfdata))
            session.commit()
            print(gfdata["name"], " INSERTED!")
        except Exception as err:
            session.rollback()
            print("# NOT INSERTED: ", err)
        print("############################################################")