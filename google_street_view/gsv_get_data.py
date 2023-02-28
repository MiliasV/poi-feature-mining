import config
import pprint
import google_streetview.api
import pandas as pd
import random
import os
import numpy as np
import cv2

# INFO
# Bounding Boxes from http://boundingbox.klokantech.com/
# Amsterdam bounding box: [4.728856,52.278139,5.06839,52.431157]
# Athens bounding box: [23.623584,37.939286,23.873866,38.059408]
# Paris bounding box: [2.224199,48.815573,2.469921,48.902145]
# London Bounding Box: [-0.351468, 51.38494, 0.148271, 51.672343]


def select_bounding_box(df, coord_list):
    # Xcoord => Longtitude
    # Ycoord => Latitude
    # coord_list = [long__low_left, lat_low_left, long_top_right, lat_top_right]
    return df.loc[(df["X"] > coord_list[0]) &
                  (df["Y"] > coord_list[1]) &
                  (df["X"] < coord_list[2]) &
                  (df["Y"] < coord_list[3])]


def create_non_exist_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

COUNTRY = "england"
CITY = "london"
COUNTRY_FOLDER = "/home/administrator/Desktop/UDS/google_street_view/downloads/" + COUNTRY
CITY_FOLDER = COUNTRY_FOLDER + "/" + CITY
POINTS_FOLDER = "/home/administrator/Desktop/UDS/geodata/" + COUNTRY + "_geodata/" + COUNTRY[0:3] + "_lat_lon_points.csv"

#create_non_exist_dir(COUNTRY_FOLDER)
#create_non_exist_dir(CITY_FOLDER)

coord_dict = {"amsterdam": [4.728856, 52.278139, 5.06839, 52.431157],
              "athens": [23.623584, 37.939286, 23.873866, 38.059408],
              "paris": [2.224199, 48.815573, 2.469921, 48.902145],
              "london": [-0.351468, 51.38494, 0.148271, 51.672343]}

df = pd.read_csv(POINTS_FOLDER)
# reduced df if needed - specific bounding box
df_red = select_bounding_box(df, coord_dict[CITY])
print(df_red.shape)
results = []
for i in range(100):
    rand = random.randint(1,df_red.shape[0])
    print("Number of Image: %d" % i)
    long = str(df_red.iloc[rand]["X"])
    lat = str(df_red.iloc[rand]["Y"])
    print(lat, long)
    for head in ["0", "90", "180", "270"]:
        params = [{'size': '600x300', 'location': lat + ',' + long,
                  'heading': head, 'pitch': '0',
                   'fov': '90', 'key': config.api_key
                 }]
        # Create a results object
        results = google_streetview.api.results(params)
        results.download_links(CITY_FOLDER)
        filename = CITY_FOLDER + "/gsv_lat_" + \
                      lat + "_long_" + \
                      long + "_head_" + \
                      head + ".jpg"
        try:
            #os.rename(CITY_FOLDER + "/gsv_0.jpg", filename)
            print("Image OK\n")
        except Exception as err:
            print(err)
            break

        save_show_img(filename, False, True, 600)

