import config
import os
import cv2
import sys
sys.path.append("..")
import pois_storing_functions
import get_map_points_to_search
import google_streetview.api
import streetview
import psycopg2


def save_show_img(filename, save_flag, show_flag, duration):
    if show_flag:
        # Load an color image in grayscale
        cv2.namedWindow(filename)  # Create a named window
        cv2.moveWindow(filename, 40, 30)  # Move it to (40,30)
        img = cv2.imread(filename)
        cv2.imshow(filename, img)
        cv2.waitKey(duration)
        cv2.destroyAllWindows()
    if not save_flag:
        os.remove(filename)


def insert_gsv_data(img, places_id, point_id, session, GSVTable, head, path):
    res = {}
    res["placesid"] = places_id
    res["id"] = res["placesid"] + "_" + img["panoid"] + "_" + "_" + head
    res["pointid"] = point_id
    res["panosid"] = img["panoid"]
    res["head"] = head
    res["year"] = img["year"]
    res["month"] = img["month"]
    res["path"] = path
    res["lat"] = img["lat"]
    res["lng"] = img["lon"]
    res["geom"] = 'POINT({} {})'.format(res["lng"], res["lat"])
    try:
        session.add(GSVTable(**res))
        session.commit()
        print("~~ ", res["pointid"], res["id"], " INSERTED!")

    except Exception as err:
        session.rollback()
        print("# NOT INSERTED: ", err)


def download_img(img, places_id, point_id, session, GSVTable):
    if "year" not in img:
        img["year"] = "None"
        img["month"] = "None"
    for head in ["0", "90", "180", "270"]:
        img_name = "row_num_" + str(point_id) + "_id_" + img[
            "panoid"] + "_head_" + head + "_lat_" + \
                   str(img["lat"]) + "_lng_" + str(img["lon"]) + "_year_" + str(img["year"])
        params = [{'size': '640x640', 'heading': head, 'pitch': '0',
                   'fov': '90', "pano": img["panoid"], "key": config.api_key}]
        # # Create a results object
        results = google_streetview.api.results(params)
        results.download_links(CITY_FOLDER)
        print(results.metadata)
        print(results.links)
        if results.metadata[0]["status"] != "ZERO_RESULTS":
            path = CITY_FOLDER + "/" + img_name
            os.rename(CITY_FOLDER + "/gsv_0.jpg", path)
            insert_gsv_data(img, places_id, point_id, session, GSVTable, head, path)



def get_points_from_db(tab, last_searched):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor()
    c.execute("SELECT rn, id, originalpointindex, lat, lng "
              "FROM "
              "(SELECT row_number() OVER (ORDER BY originalpointindex NULLS LAST) AS rn,  "
              "id, originalpointindex, lat, lng "
              " FROM {table} ) AS rowselection "
              "WHERE rn>={last_row}".format(table=tab, last_row=last_searched))
    return c


def get_missed_points_from_db(tab, id_list):
    conn = psycopg2.connect(database="pois", user="postgres", password="postgres")
    c = conn.cursor()
    c.execute("SELECT rn, id, originalpointindex, lat, lng "
              "FROM "
              "(SELECT row_number() OVER (ORDER BY originalpointindex NULLS LAST) AS rn,  "
              "id, originalpointindex, lat, lng "
              " FROM {table} ) AS rowselection "
              "WHERE id IN {list}".format(table=tab, list=id_list))
    return c


if __name__ == "__main__":
        try:
            city = "ams"
            source = "gsv"
            CITY_FOLDER = "/home/bill/Desktop/thesis/code/UDS/google_street_view/images/" + city
            #c, rad, logfile, errorfile = get_map_points_to_search.config_parameters_for_searching("gsv")
            logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_whole_clipped_point_iterations.txt"
            not_inserted_file = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_whole_clipped_not_inserted.txt"
            last_searched_id = pois_storing_functions.get_last_id_from_logfile(logfile)
            points = get_points_from_db("google_ams_whole_clipped_40", last_searched_id)
            with open(not_inserted_file, "r") as f:
                lines = f.read().splitlines()

            session, GSVTable = pois_storing_functions.setup_db("gsv_ams_whole_clipped_40",
                                                                 "gsv_ams_whole_clipped_count", "gsv")

            # For each point --> search nearby in google
            for row_number, places_id, point_id, point_lat, point_lng in points:
                get_map_points_to_search.log_last_searched_point(logfile, row_number)
                found_flag = False
                print("POINT: ", row_number, places_id, point_id, point_lat, point_lng)
                panoids = streetview.panoids(lat=point_lat, lon=point_lng)
                if not panoids:
                    with open(not_inserted_file, "a") as text_file:
                        print(f"Not found for point \n row number: {row_number}, point_id: {point_id}, "
                              f"place id: {places_id}", file=text_file)
                        text_file.close()
                        continue
                print(panoids)
                # For only last year's images
                # max_year_panoid = max(panoids, key=lambda x : x["year"] if "year" in x.keys() else 0)
                # print(max_year_panoid)
                # download_img(max_year_panoid, places_id, point_id, session, GSVTable,
                for img in panoids:
                    if "year" in img:
                        print(img)
                        download_img(img, places_id, point_id, session, GSVTable)
                        found_flag = True
                # if year information not found
                if not found_flag and panoids:
                    print("o")
                    download_img(panoids[0], row_number, point_id, session, GSVTable)
            # break
        except Exception as err:
            print(err)
            continue