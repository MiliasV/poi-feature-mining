# Master Thesis Project
## Title:Extraction and combination of multidimensional point-of-interest features for the classification of urban place types
##### Author: Vasileios Milias
##### University: TU Delft
##### Department: Web Information Systems, Social Data Lab

### General

This code was developed during my MSc thesis project (see abstract below). It includes code for collecting data from Google, Google  Street View, Foursquare and Twitter, for extracting various types of features from those data and for predicting the points-of-interest (POIs) types. It definitely needs polishing as in various parts it is quite "quick and dirty".  

Don't hesitate to contact me for any questions.

### Abstract

The digital representations of places, known as Points-Of-Interest (POIs), have been the core element of various studies and platforms such as online mapping services (e.g. Google Maps) and location based social networks (e.g. Foursquare). The use of POIs as proxies of the real-world-places facilitates the study of places, urban environments and, consequently,  human behavior. Therefore, the extent to which the POIs manage to capture the complex multidimensional nature of physical places defines the limits of all those platforms and of humans' essential understanding of places.

Admittedly, the already existing POI data sources tend to represent differently the physical places  (e.g. focus on specific aspects of places) and their data are being produced in a variety of ways (e.g. user generated data or non-user generated data). In addition, multiple sources exist that indirectly include place-related information as, for instance, Google Street View  which contains images of the exterior of places without providing a direct link between the image and the corresponding place-entity. Thus, an interesting challenge arises which is how could all those diverse data coming from different data sources be combined towards the creation of a better digital representation of places.

This thesis introduces an innovative approach to the extraction and combination of multidimensional POI features from various place-related data sources towards the study of urban places. It consists of two main parts: (1)  the process of selecting, extracting and combining multidimensional POI features from various  sources which reflect the high dimensional nature of places and (2) the use of the extracted features to discover  which of those - and to what extent - better define and distinguish urban places in respect to their core characteristic, their main function. 

Regarding the first part, for the combination of POI data sources a "matching" algorithm is developed whose goal is the identification of POIs which belong to different POI data sources and represent the same physical place and is based on the comparison of a set of attributes such as location, name and website. 
For the extraction of the POI features  the need of specialized techniques  according to the nature of the different data is revealed and several methods are discussed and used. 

The second part concentrates in data collected from two capitals, Amsterdam and Athens. A machine learning classifier is trained on different combinations of features extracted from those data and their importance for distinguishing the urban place types is computed and compared. The results,  among other, support that the functional  (e.g. opening/closing times) and  experiential characteristics (e.g. topics extracted from reviews) are the strongest indicators of a place's type  independently of the context (e.g. city) while the exterior visual appearance of places does not provide such valuable information. The combination of the extracted features lead to an F1-score of around 60\% when classifying POIs by their type among 10 classes (multiclass problem) and around 90\% when predicting if a POI is of a certain type or not (binary problem). 

Overall, the importance of combining multiple data sources in order to capture the complex nature of places is successfully supported by the results and the  features that tend to better "describe" places in respect to their main function are discovered and further explored.  
