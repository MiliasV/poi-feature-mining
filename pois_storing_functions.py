from sqlalchemy.ext.automap import automap_base
from geoalchemy2 import Geometry
from sqlalchemy import *
from sqlalchemy.orm import Session


def create_fsq_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("street", String),
              Column("streetnum", String),
              Column("checkins", Numeric),
              Column("userscount", Numeric),
              Column("tipcount", Numeric),
              Column("visitscount", Numeric),
              Column("rating", Numeric),
              Column("price", String),
              Column("likescount", Numeric),
              Column("photoscount", Numeric),
              Column("website", String),
              Column("facebook", String),
              Column("twitter", String),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_fsq_matched_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", Numeric),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("street", String),
              Column("streetnum", String),
              Column("checkins", Numeric),
              Column("userscount", Numeric),
              Column("tipcount", Numeric),
              Column("visitscount", Numeric),
              Column("rating", Numeric),
              Column("price", String),
              Column("likescount", Numeric),
              Column("photoscount", Numeric),
              Column("website", String),
              Column("facebook", String),
              Column("twitter", String),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_google_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("streetlng", String),
              Column("streetsrt", String),
              Column("streetnumlng", String),
              Column("streetnumsrt", String),
              Column("website", String),
              Column("rscount", Numeric),
              Column("phcount", Numeric),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("rating", String),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_google_matched_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", Numeric),
              Column("originalpointindex", Numeric),
              Column("name", String),
              Column("type", String),
              Column("type1", String),
              Column("type2", String),
              Column("type3", String),
              Column("type4", String),
              Column("type5", String),
              Column("streetlng", String),
              Column("streetsrt", String),
              Column("streetnumlng", String),
              Column("streetnumsrt", String),
              Column("website", String),
              Column("rscount", Numeric),
              Column("phcount", Numeric),
              Column("phone", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("searchlat", Numeric),
              Column("searchlng", Numeric),
              Column("rating", String),
              Column("poptimes", String),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_gsv_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("pointid", Numeric),
              Column("placesid", String),
              Column("panosid", String),
              Column("head", String),
              Column("year", String),
              Column("month", String),
              Column("path", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("geom", Geometry('Point')))
        metadata.create_all()


def create_twitter_pois_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("pointid", Numeric),
              Column("fsqid", String),
              Column("lang", String),
              Column("createdat", String),
              Column("year", String),
              Column("month", String),
              Column("day", String),
              Column("hour", String),
              Column("lang", String),
              Column("text", String),
              Column("favoritecount", Numeric),
              Column("retweetcount", Numeric),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("geom", Geometry('Point')),
              Column("json", String))
        metadata.create_all()


def create_matching_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", String),
              Column("fsqid", String),
              Column("googleid", String),
              Column("reason", String)
              )
        metadata.create_all()


def create_scene_features_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", String),
              Column("type", String),
              Column("placesid", String),
              Column("panosid", String),
              Column("head", String),
              Column("typeofenvironment", String),
              Column("scene1", String),
              Column("scene1prob", Float),
              Column("scene2", String),
              Column("scene2prob", Float),
              Column("scene3", String),
              Column("scene3prob", Float),
              Column("scene4", String),
              Column("scene4prob", Float),
              Column("sceneattr1", String),
              Column("sceneattr2", String),
              Column("sceneattr3", String),
              Column("sceneattr4", String),
              Column("sceneattr5", String),
              Column("sceneattr6", String),
              Column("sceneattr7", String),
              Column("sceneattr8", String),
              Column("sceneattr9", String),
              Column("year", Numeric),
              Column("month", Numeric),
              Column("path", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("geom", Geometry('Point')))
        metadata.create_all()


def create_object_detection_coco_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", String),
              Column("placesid", String),
              Column("type", String),
              Column("panosid", String),
              Column("head", String),
              Column("person", Numeric),
              Column("personcount", Numeric),
              Column("personhighestprob", Numeric),
              Column("bicycle", Numeric),
              Column("bicyclecount", Numeric),
              Column("bicyclehighestprob", Numeric),
              Column("car", Numeric),
              Column("carcount", Numeric),
              Column("carhighestprob", Numeric),
              Column("motorcycle", Numeric),
              Column("motorcyclecount", Numeric),
              Column("motorcyclehighestprob", Numeric),
              Column("bus", Numeric),
              Column("buscount", Numeric),
              Column("bushighestprob", Numeric),
              Column("train", Numeric),
              Column("traincount", Numeric),
              Column("trainhighestprob", Numeric),
              Column("truck", Numeric),
              Column("truckcount", Numeric),
              Column("truckhighestprob", Numeric),
              Column("boat", Numeric),
              Column("boatcount", Numeric),
              Column("boathighestprob", Numeric),
              Column("trafficlight", Numeric),
              Column("trafficlightcount", Numeric),
              Column("trafficlighthighestprob", Numeric),
              Column("firehydrant", Numeric),
              Column("firehydrantcount", Numeric),
              Column("firehydranthighestprob", Numeric),
              Column("stopsign", Numeric),
              Column("stopsigncount", Numeric),
              Column("stopsignhighestprob", Numeric),
              Column("bench", Numeric),
              Column("benchcount", Numeric),
              Column("benchhighestprob", Numeric),
              Column("pottedplant", Numeric),
              Column("pottedplantcount", Numeric),
              Column("pottedplanthighestprob", Numeric),
              Column("year", Numeric),
              Column("month", Numeric),
              Column("path", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("geom", Geometry('Point')))
        metadata.create_all()


def create_object_detection_oid_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("point", String),
              Column("placesid", String),
              Column("type", String),
              Column("panosid", String),
              Column("head", String),
              Column("person", Numeric),
              Column("personcount", Numeric),
              Column("personhighestprob", Numeric),
              Column("clothing", Numeric),
              Column("clothingcount", Numeric),
              Column("clothinghighestprob", Numeric),
              Column("man", Numeric),
              Column("mancount", Numeric),
              Column("manhighestprob", Numeric),
              Column("woman", Numeric),
              Column("womancount", Numeric),
              Column("womanhighestprob", Numeric),
              Column("tree", Numeric),
              Column("treecount", Numeric),
              Column("treehighestprob", Numeric),
              Column("houseplant", Numeric),
              Column("houseplantcount", Numeric),
              Column("houseplanthighestprob", Numeric),
              Column("flower", Numeric),
              Column("flowercount", Numeric),
              Column("flowerhighestprob", Numeric),
              Column("building", Numeric),
              Column("buildingcount", Numeric),
              Column("buildinghighestprob", Numeric),
              Column("skyscraper", Numeric),
              Column("skyscrapercount", Numeric),
              Column("skyscraperhighestprob", Numeric),
              Column("house", Numeric),
              Column("housecount", Numeric),
              Column("househighestprob", Numeric),
              Column("conveniencestore", Numeric),
              Column("conveniencestorecount", Numeric),
              Column("conveniencestorehighestprob", Numeric),
              Column("office", Numeric),
              Column("officecount", Numeric),
              Column("officehighestprob", Numeric),
              Column("streetlight", Numeric),
              Column("streetlightcount", Numeric),
              Column("streetlighthighestprob", Numeric),
              Column("trafficlight", Numeric),
              Column("trafficlightcount", Numeric),
              Column("trafficlighthighestprob", Numeric),
              Column("trafficsign", Numeric),
              Column("trafficsigncount", Numeric),
              Column("trafficsignhighestprob", Numeric),
              Column("tent", Numeric),
              Column("tentcount", Numeric),
              Column("tenthighestprob", Numeric),
              Column("vehicle", Numeric),
              Column("vehiclecount", Numeric),
              Column("vehiclehighestprob", Numeric),
              Column("landvehicle", Numeric),
              Column("landvehiclecount", Numeric),
              Column("landvehiclehighestprob", Numeric),
              Column("car", Numeric),
              Column("carcount", Numeric),
              Column("carhighestprob", Numeric),
              Column("bike", Numeric),
              Column("bikecount", Numeric),
              Column("bikehighestprob", Numeric),
              Column("boat", Numeric),
              Column("boatcount", Numeric),
              Column("boathighestprob", Numeric),
              Column("year", Numeric),
              Column("month", Numeric),
              Column("path", String),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("geom", Geometry('Point')))
        metadata.create_all()


def create_count_per_poi_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", Numeric, primary_key=True, nullable=False),
              Column("countplaces", Numeric),
              Column("countduplicates", Numeric)
              )
        metadata.create_all()


def create_text_features_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("name", String),
              Column("point", Numeric),
              Column("lat", Numeric),
              Column("lng", Numeric),
              Column("type", String),
              Column("timediffavg", Numeric),
              Column("timediffmedian", Numeric),
              Column("topiceng51", Numeric),
              Column("topicnl51", Numeric),
              Column("topic51", Numeric),
              Column("topiceng52", Numeric),
              Column("topicnl52", Numeric),
              Column("topic52", Numeric),
              Column("topiceng53", Numeric),
              Column("topicnl53", Numeric),
              Column("topic53", Numeric),
              Column("topiceng54", Numeric),
              Column("topicnl54", Numeric),
              Column("topic54", Numeric),
              Column("topiceng55", Numeric),
              Column("topicnl55", Numeric),
              Column("topic55", Numeric),
              Column("topiceng101", Numeric),
              Column("topicnl101", Numeric),
              Column("topic101", Numeric),
              Column("topiceng102", Numeric),
              Column("topicnl102", Numeric),
              Column("topic102", Numeric),
              Column("topiceng103", Numeric),
              Column("topicnl103", Numeric),
              Column("topic103", Numeric),
              Column("topiceng104", Numeric),
              Column("topicnl104", Numeric),
              Column("topic104", Numeric),
              Column("topiceng105", Numeric),
              Column("topicnl105", Numeric),
              Column("topic105", Numeric),
              Column("topiceng106", Numeric),
              Column("topicnl106", Numeric),
              Column("topic106", Numeric),
              Column("topiceng107", Numeric),
              Column("topicnl107", Numeric),
              Column("topic107", Numeric),
              Column("topiceng108", Numeric),
              Column("topicnl108", Numeric),
              Column("topic108", Numeric),
              Column("topiceng109", Numeric),
              Column("topicnl109", Numeric),
              Column("topic109", Numeric),
              Column("topiceng1010", Numeric),
              Column("topicnl1010", Numeric),
              Column("topic1010", Numeric),
              Column("entweetcount", Numeric),
              Column("nltweetcount", Numeric),
              Column("totaltweetcount", Numeric),
              Column("enwordcount", Numeric),
              Column("nlwordcount", Numeric),
              Column("totalwordcount", Numeric),
              Column("engavgword", Numeric),
              Column("nlavgword", Numeric),
              Column("avgword", Numeric),
              Column("enpolpoly", Numeric),
              Column("nlpolpoly", Numeric),
              Column("enpolblob", Numeric),
              Column("ensubjblob", Numeric),
              Column("nlpolblob", Numeric),
              Column("nlsubblob", Numeric)
              )
        metadata.create_all()


def create_similarities_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("match", Numeric),
              Column("namerosimilarity", Numeric),
              Column("namedlevensimilarity", Numeric),
              Column("namelevensimilarity", Numeric),
              Column("namephoneticsimilarity", Numeric),
              Column("namelenlongsubstring", Numeric),
              Column("websiterosimilarity", Numeric),
              Column("websitedlevensimilarity", Numeric),
              Column("websitelevensimilarity", Numeric),
              Column("websitephoneticsimilarity", Numeric),
              Column("websitelenlongsubstring", Numeric),
              Column("phonerosimilarity", Numeric),
              Column("phonedlevensimilarity", Numeric),
              Column("phonelevensimilarity", Numeric),
              Column("phonephoneticsimilarity", Numeric),
              Column("phonelenlongsubstring", Numeric),
              Column("streetrosimilarity", Numeric),
              Column("streetdlevensimilarity", Numeric),
              Column("streetlevensimilarity", Numeric),
              Column("streetphoneticsimilarity", Numeric),
              Column("streetlenlongsubstring", Numeric),
              Column("streetnumrosimilarity", Numeric),
              Column("streetnumdlevensimilarity", Numeric),
              Column("streetnumlevensimilarity", Numeric),
              Column("streetnumphoneticsimilarity", Numeric),
              Column("streetnumlenlongsubstring", Numeric)
              )
        metadata.create_all()


def create_gf_features_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("name", String),
              Column("type", String),
              Column("point", Numeric),
              Column("gid", String),
              Column("fid", String),
              Column("ghasweb", Numeric),
              Column("fhasweb", Numeric),
              Column("ghasphone", Numeric),
              Column("fhasphone", Numeric),
              Column("grscount", Numeric),
              Column("grating", Numeric),
              Column("frating", Numeric),
              Column("ftipcount", Numeric),
              Column("fprice", String),
              Column("fphotoscount", Numeric),
              Column("flikescount", Numeric),
              Column("fhasfacebook", Numeric),
              Column("fhastwitter", Numeric),
              Column("ghaspoptimes", Numeric),
              Column("Monday", String),
              Column("Tuesday", String),
              Column("Wednesday", String),
              Column("Thursday", String),
              Column("Friday", String),
              Column("Saturday", String),
              Column("Sunday", String),
              Column("day0close", String),
              Column("day0open", String),
              Column("day1close", String),
              Column("day1open", String),
              Column("day2close", String),
              Column("day2open", String),
              Column("day3close", String),
              Column("day3open", String),
              Column("day4close", String),
              Column("day4open", String),
              Column("day5close", String),
              Column("day5open", String),
              Column("day6close", String),
              Column("day6open", String)
              )
        metadata.create_all()


def create_reviews_features_table(engine, table_name, metadata):
    # if table does not exist
    if not engine.dialect.has_table(engine, table_name):
        Table(table_name, metadata,
              Column("id", String, primary_key=True, nullable=False),
              Column("gid", String),
              Column("name", String),
              Column("type", String),
              Column("point", Numeric),
              Column("lang", String),
              Column("text", String),
              Column("processedldatext", String)
              )
        metadata.create_all()

class ALTable(object):
    pass


def setup_db(pois_table_name, count_table_name, source):
    Base = automap_base()
    # Connect to the database
    #db = create_engine("sqlite:///../../../databases/google_places.db")
    db = create_engine('postgresql://postgres:postgres@localhost/pois')
    # create object to manage table definitions
    metadata = MetaData(db)
    # create table if it doesn't exist - also define the table.
    # change it in GTable as well!
    if source == "google":
        create_google_pois_table(db, pois_table_name, metadata)
    elif source == "matching_table":
        create_matching_table(db, pois_table_name, metadata)
    elif source == "FSQ":
        create_fsq_pois_table(db, pois_table_name, metadata)
        Base.prepare(db, reflect=True)
        STable = getattr(Base.classes, pois_table_name)
        #CTable = getattr(Base.classes, count_table_name)
        # create a Session
        session = Session(db)
        return session, STable, []

    elif source == "fsq_matched":
        create_fsq_matched_pois_table(db, pois_table_name, metadata)

    elif source == "google_matched":
        create_google_matched_pois_table(db, pois_table_name, metadata)

    elif source == "twitter":
        create_twitter_pois_table(db, pois_table_name, metadata)
        # Base.prepare(db, reflect=True)
        # TTable = getattr(Base.classes, pois_table_name)
        # # create a Session
        # session = Session(db)
        # return session, TTable
    elif source == "gsv":
        create_gsv_pois_table(db, pois_table_name, metadata)
        # Base.prepare(db, reflect=True)
        # STable = getattr(Base.classes, pois_table_name)
        # # create a Session
        # session = Session(db)
        # return session, STable
    elif source == "scene_features":
        create_scene_features_table(db, pois_table_name, metadata)
        # Base.prepare(db, reflect=True)
        # STable = getattr(Base.classes, pois_table_name)
        # # create a Session
        # session = Session(db)
        # return session, STable
    elif source == "text_features":
        create_text_features_table(db, pois_table_name, metadata)
    elif source == "similarities":
        create_similarities_table(db, pois_table_name, metadata)
    elif source == "od_coco":
        create_object_detection_coco_table(db, pois_table_name, metadata)
    elif source == "od_oid":
        create_object_detection_oid_table(db, pois_table_name, metadata)
    elif source == "gf":
        create_gf_features_table(db, pois_table_name, metadata)
    elif source == "reviews":
        create_reviews_features_table(db, pois_table_name, metadata)
    else:
        print("ERROR, none of the right options were given")
        return 0
    #create_count_per_poi_table(db, count_table_name, metadata)
    # reflect the tables
    Base.prepare(db, reflect=True)
    STable = getattr(Base.classes, pois_table_name)
    #CTable = getattr(Base.classes, count_table_name)
    #data_table = Table('google_ams_centroids_40', metadata, autoload=True)
    # create a Session
    session = Session(db)
    return session, STable #, CTable


def get_last_id_from_logfile(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()
        f.close()
        return int(lines[-1])


def insert_count_data(session, CTable, poi_number, count_places, count_dupl):
    try:
        session.add(CTable(id=poi_number, countplaces=count_places, countduplicates=count_dupl))
        session.commit()
    except Exception as err:
        session.rollback()
        print("# [COUNT] NOT INSERTED: ", err)