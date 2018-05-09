import sqlite3


def get_last_id_from_logfile(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()
        f.close()
        return int(lines[-1])


def config_parameters_for_searching(source):
    city = "ams"
    rad = 20
    # databases
    logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_point_iterations.txt"
    center_db = "/home/bill/Desktop/thesis/maps/" + city + "_center_centroids_40.sqlite"
    # connect to osm db
    conn = sqlite3.connect(center_db)
    c = conn.cursor()
    last_searched_id = get_last_id_from_logfile(logfile)
    c.execute("SELECT ogc_fid, lat, Lon "
              "FROM ams_center_centroids_40 "
              "WHERE id>={last_id}-1".format(last_id=last_searched_id))
    return c, rad, logfile


def log_last_searched_point(logfile, ogc_fid):
    with open(logfile, "w") as text_file:
        print(f"Last searched point \n{ogc_fid}", file=text_file)
        text_file.close()