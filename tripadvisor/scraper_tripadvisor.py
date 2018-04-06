import scrapy


# CITY = {ath OR ams)
CITY = "ath"

# replace '*****' with query place
search_ath_url = "https://www.tripadvisor.com/Search?geo=189400&latitude=&longitude=&searchNearby=&pid=3826&redirect=&startTime=1522600584446&uiOrigin=MASTHEAD&q=*****&supportedSearchTypes=find_near_stand_alone_query&enableNearPage=true&returnTo=__2F__&searchSessionId=43C3DADE034542F2B0F3DE99BAE7C1781522600563687ssid#&ssrc=a&o=0"
search_ams_url = "https://www.tripadvisor.com/Search?geo=188590&latitude=&longitude=&searchNearby=&pid=3826&redirect=&startTime=1522600197799&uiOrigin=MASTHEAD&q=*****&supportedSearchTypes=find_near_stand_alone_query&enableNearPage=true&returnTo=__2F__&searchSessionId=43C3DADE034542F2B0F3DE99BAE7C1781522600180776ssid"

pois_list = ["5+dromoi"]#, "A+for+athens", "Jazz Bar"]


class TripAdvisorSpider(scrapy.Spider):
    name = 'tripadvisor_spider'

    def start_requests(self):
        urls = []
        if CITY == "ath":
            search_url = search_ath_url
        else:
            search_url = search_ams_url

        for poi in pois_list:
            urls.append(search_url.replace("*****", poi))

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_search_results)

    def parse_search_results(self, response):
        # d = {'name':response.xpath('//body//div[@class="title"]//a/text())').extract()}
        extracted_info = []
        # select name
        # select address
        # select link for more details
        for node in response.xpath('//body//div[@class="title"]//span//text() |'
                                   '//body//div[@class="title"]/@onclick'):
                                   #'//body//div[@class="type"]//span[contains(@class,"ui_icon")]/@class'):
            #print(node.extract())
            extracted_info.append(node.extract())

        # create tuple with the above values --> (link, name)
        places_addr = list(zip(extracted_info[0::2], extracted_info[1::2])) #, extracted_info[2::3]))
        for i in places_addr[0:5]:
            print("################################################")
            print(i)
            url = i[0].split(",")[-1][2:-2]
            name = i[1]
            ## ---> check name and if there is similarity continue !!!!

            # print("*****", name, type, url)
            yield scrapy.Request(url="https://www.tripadvisor.com" + url, callback=self.parse_place)

    def parse_place(self, response):
        place_info = []
        # get name, address,
        for node in response.xpath('//body//h1[@class="heading_title"]//text() |'
                                   '//body//div[contains(@class, "clickable")]//span[@class="street-address"]//text()'):
            #print(node.extract())
            place_info.append(node.extract())
        # for i in place_info:
        #     print("****", i)
         #print(place_info)
