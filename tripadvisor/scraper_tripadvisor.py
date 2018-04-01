import scrapy

# replace ***** with query place
search_ath_url = "https://www.tripadvisor.com/Search?geo=189400&latitude=&longitude=&searchNearby=&pid=3826&redirect=&startTime=1522600584446&uiOrigin=MASTHEAD&q=*****&supportedSearchTypes=find_near_stand_alone_query&enableNearPage=true&returnTo=__2F__&searchSessionId=43C3DADE034542F2B0F3DE99BAE7C1781522600563687ssid#&ssrc=a&o=0"
search_ams_url = "https://www.tripadvisor.com/Search?geo=188590&latitude=&longitude=&searchNearby=&pid=3826&redirect=&startTime=1522600197799&uiOrigin=MASTHEAD&q=*****&supportedSearchTypes=find_near_stand_alone_query&enableNearPage=true&returnTo=__2F__&searchSessionId=43C3DADE034542F2B0F3DE99BAE7C1781522600180776ssid"

pois_list = ["5+dromoi", "A+for+athens", "Jazz Bar"]


class TripAdvisorSpider(scrapy.Spider):
    name = 'tripadvisor_spider'
    #start_urls = ['https://www.tripadvisor.com/Restaurants-g189400-Athens_Attica.html']
    start_urls = ['https://www.tripadvisor.com/Search?geo=189400&latitude=&longitude=&searchNearby=&pid=3826&redirect=&startTime=1522414764824&uiOrigin=MASTHEAD&q=5+dromoi&supportedSearchTypes=find_near_stand_alone_query&enableNearPage=true&returnTo=__2F__Restaurants__2D__g189400__2D__Athens__5F__Attica__2E__html&searchSessionId=43C3DADE034542F2B0F3DE99BAE7C1781522414749772ssid#&ssrc=a&o=0']


    def parse(self, response):
        # d = {'name':response.xpath('//body//div[@class="title"]//a/text())').extract()}
        extracted_info = []
        # select name
        # select address
        # select link for more details
        for node in response.xpath('//body//div[@class="title"]//span/text() |'
                                   '//body//div[@class="address"]/text() |'
                                   '//body//div[@class="title"]/@onclick'):
            print(node.extract())
            extracted_info.append(node.extract())
        print(extracted_info)
        # create tuple with the above values
        places_addr = list(zip(extracted_info[0::3], extracted_info[1::3], extracted_info[2::3]))
        for i in places_addr:
            print(i[1].split(",")[-1][2:-2],"--->", i[0], " ----->0", i[2])
            #print(i)
