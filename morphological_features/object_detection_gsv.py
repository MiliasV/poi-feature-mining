import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys
sys.path.append("/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/")
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import postgis_functions
import collections
import pois_storing_functions
import logging_functions
#cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.
#from utils import label_map_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

######################
# Model preparation  #
#####################
#'ava_label_map_v2.1.pbtxt'
# rfcn_resnet101_coco_2018_01_28
#'faster_rcnn_resnet101_ava_v2.1_2018_04_30'
# rfcn_resnet101_coco_2018_01_28

########
# COCO #
########
# labels => mscoco_label_map.pbtxt

# 1) Accurate - 43
# rcnn_nas_coco_2018_01_28 (model with highest mAP but slowest)

# 2) Accurate - 37
# faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28

# 3) Accurate - 32
# faster_rcnn_resnet101_coco_2018_01_28 (way faster than (2) ) USED ONE

# 4) Fastest - 21
# ssd_mobilenet_v1_coco_2017_11_17

########
# OID #
########
# labels => oid_bbox_trainable_label_map.pbtxt

# 1) Accurate - 37
# faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
# 2) Accurate - < 37
# faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28 (one used!!!)


################################
# VARIABLES DEPENDING ON MODEL #
################################
TYPE = "coco"
city = "ams"

cc_category_ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 64]
oid_category_ids = [1, 2, 3, 7, 5, 75, 17, 9, 46, 20, 188, 114, 69, 190, 144, 132, 8, 10, 14, 43, 36]
if TYPE == "coco":
    category_ids = cc_category_ids
    MODEL_NAME = "faster_rcnn_resnet101_coco_2018_01_28"
    LABELS = 'mscoco_label_map.pbtxt'
    TABLE = "matched_od_coco_" + city
else:
    category_ids = oid_category_ids
    MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28"
    LABELS = 'oid_bbox_trainable_label_map.pbtxt'
    TABLE = "matched_od_oid_" + city


####################################################################################

MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

REL_PATH = "/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/"
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = REL_PATH  + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(REL_PATH + '/object_detection/data', LABELS)

NUM_CLASSES = 600000

# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#opener.retrieve(REL_PATH + MODEL_FILE, MODEL_FILE)

tar_file = tarfile.open(REL_PATH + MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, REL_PATH)#os.getcwd())
# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
if __name__ == '__main__':
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    #source = "coco"
    session, ODTable = pois_storing_functions.setup_db(TABLE, "notused", TYPE)
    # selected categories' ids
    # cc_categories = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    #                  "boat", "trafficlight", "firehydrant", "stopsign", "bench", "pottedplant"]
    # this should change depending on the labels
    sel_categories = {x:category_index[x] for x in category_ids}
    #logfile = "/home/bill/Desktop/thesis/logfiles/" + source + "_" + city + "_matched.txt"
    #last_searched_id = logging_functions.get_last_id_from_logfile(logfile)
    #imgs = postgis_functions.get_photos_from_db("matched_gsv_ams", last_searched_id)#last_searched_id)
    imgs = postgis_functions.get_photos_for_od("matched_places_gsv_" + city, TABLE)

    # for i in sel_categories:
    #     print('Column("{name}", Numeric),'.format(name=sel_categories[i]["name"].lower().replace(" ", "")))
    #     print('Column("{count}", Numeric),'.format(count=sel_categories[i]["name"].lower().replace(" ", "")+"count"))
    #     print('Column("{high}", Numeric),'.format(high=sel_categories[i]["name"].lower().replace(" ", "")+"highestprob"))
    # print(a)
    examples = ["/home/bill/Desktop/gsv_examples/row_num_207793_id_kApySdDCXlQDy-NNgE_jQg_head_0_lat_52.37113663706489_lng_4.891840962872775_year_2016",
                "/home/bill/Desktop/gsv_examples/row_num_157634_id_rgmIZ8FDeQBz5Hp_fzQEPw_head_90_lat_52.38490205676492_lng_4.853863636798268_year_2014"]
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for img in imgs:
            #while True:
                print(img)
                image_np = cv2.imread(img["path"])
                #r, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                # print(scores)
                # print(classes)
                # indices for prob > 0.3
                detected_index = np.where(scores>0.3)
                # print(detected_index)
                # detected classes
                detected_classes = classes[detected_index]
                detected_scores = scores[detected_index]
                # detected scores
                counts = collections.Counter(classes[detected_index])
                # print(sel_categories)
                # print(counts)
                # print(detected_scores)
                # print(detected_classes)
                for i, cl in enumerate(detected_classes):
                    # print(i, cl)
                    if cl in sel_categories:
                        img[sel_categories[cl]["name"].replace(" ","").lower()] = 1
                        img[sel_categories[cl]["name"].replace(" ","").lower() + "count"] = counts[cl]
                        img[sel_categories[cl]["name"].replace(" ","").lower() + "highestprob"] = float(detected_scores[i])
                #img.pop("rn", None)
                # get place of image from db
                gpoint = postgis_functions.get_matchid_by_id("matched_places_google_" + city, img["placesid"])
                # find matched fsq place
                fplace = postgis_functions.get_matched_poi("matched_places_fsq_" + city, gpoint)
                # get type of place
                ftype = postgis_functions.get_type_of_place(fplace)
                img["type"] = ftype
                # print(ftype)
                try:
                    #logging_functions.log_last_searched_point(logfile, img["point"])
                    session.add(ODTable(**img))
                    session.commit()
                    print(img["placesid"], " INSERTED!")
                except Exception as err:
                    session.rollback()
                    print("# NOT INSERTED: ", err)

                # #Visualization of the results of a detection.
                # print(classes)
                # print(category_index)
                # print(classes)
                # print(scores)
                # print(category_index)
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     image_np,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=3)
                #
                # cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                # cv2.waitKey(400000000)
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()