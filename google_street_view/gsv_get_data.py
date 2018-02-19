import config
import pprint
import google_streetview.api
import pandas as pd
import random
import os


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


COUNTRY = "greece"
CITY = "athens"
COUNTRY_FOLDER = "/home/administrator/Desktop/UDS/google_street_view/downloads/" + COUNTRY
CITY_FOLDER = COUNTRY_FOLDER + "/" + CITY
POINTS_FOLDER = "/home/administrator/Desktop/UDS/" + COUNTRY + "_geodata/" + COUNTRY[0:3] + "_lat_lon_points.csv"

print(POINTS_FOLDER)
create_non_exist_dir(COUNTRY_FOLDER)
create_non_exist_dir(CITY_FOLDER)

coord_dict = {"amsterdam": [4.728856, 52.278139, 5.06839, 52.431157],
              "athens": [23.623584, 37.939286, 23.873866, 38.059408]}
# ATHENS_COORD = [23.623584,37.939286,23.873866,38.059408]
# AMSTERDAM_COORD = [4.728856,52.278139,5.06839,52.431157]
# Amsterdam bounding box: 4.728856,52.278139,5.06839,52.431157
# Athens bounding box: 23.623584,37.939286,23.873866,38.059408


df = pd.read_csv(POINTS_FOLDER)

# reduced df if needed- specific bounding box
df_red = select_bounding_box(df, coord_dict[CITY])
results = []
#print(df_red.shape)
for i in range(500): #df_red.shape[0]):
    print("Number of Image: %d" % i)
    #print(num_points)
    #rand = random.randint(1, df_red.shape[0])
    long = str(df_red.iloc[i]["X"])
    lat = str(df_red.iloc[i]["Y"])

    print(long,lat)
    #pp = pprint.PrettyPrinter(indent=4)
    for head in ["0", "90", "180", "270"]:
        # Define parameters for street view api
        # location: latitude, longtitude
        params = [{
          'size': '600x300', # max 640x640 pixels
          'location': lat + ',' + long,
          'heading': head,
          'pitch': '0',
          'fov': '90',
          'key': config.api_key
        }]

        # Create a results object

        results = google_streetview.api.results(params)
        #print(results.metadata)
        #results.metadata["_file"] = "gsv_lat_" + lat + "_long_" + long + ".jpg"
        # Download images to directory 'downloads'
        results.download_links(CITY_FOLDER)
        try:
            os.rename(CITY_FOLDER + "/gsv_0.jpg",
                      CITY_FOLDER + "/gsv_lat_" +
                      lat + "_long_" +
                      long + "_head_" +
                      head + ".jpg")
            print("Image OK\n")
        except Exception as err:
            print(err)
            break
        # pp.pprint(results.metadata)

