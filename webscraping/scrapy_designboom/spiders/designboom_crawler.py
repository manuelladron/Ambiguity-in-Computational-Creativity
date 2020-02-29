# -*- coding: utf-8 -*-
import scrapy
from scrapy_designboom.items import ScrapyDesignboomItem
import pdb

class Designboom(scrapy.Spider):

    # Spider's name. It is for internal reference, and we use it to make the spider crawl to do the webscraping when calling it
    # in terminal the following command:
    # -----> scrapy crawl designboom_crawler -o outfile_name.json <-------
    name = 'designboom_crawler'

    # ART
    # allowed_domains = ['designboom.com/art']
    # start_urls = ['http://designboom.com/art/']

    # DESIGN
    # allowed_domains = ['designboom.com/design']
    # start_urls = ['http://designboom.com/design/']

    # ARCHITECTURE
    # this is automatically generated when we start a new scrapy project, this is being take care of. But, like in this
    # example, where I am scrapping data from different sections of the website, we just use this file and change this
    # starting urls.
    # The start_url is the initial page where the spider starts crawling.
    allowed_domains = ['designboom.com/architecture']
    start_urls = ['https://www.designboom.com/architecture']

    # this init I created just to differentiate the first page from the rest. The very first page doesnt have a "previous
    # page", so it needs a different xpath.

    def __init__(self):
        self.counter = 0

    def parse(self, response):
        # DEPTH 270 ART

        #this is to help debugging. I manually looked which page number is the last one.
        if response.meta['depth'] > 675:
            print('Loop?')


        # NOT USING THIS, BUT WE CAN INTIATE THE ITEM (the main structure where we store the data) HERE, IF WE NEED TO
        # PASS SOME INFORMATION FROM THE MAIN PAGE.
        # WE ARE NOT USING THIS BECAUSE ANY INFORMATION COMES FROM CLICKING ON EACH SPECIFIC ENTRY OF THE WEBSITE.
        # Designboom has 28 entries per page. We individually entry in each of them. Say that you wanted to get the image
        # it shows in the main page, we would need to put the image in the ITEM here. But we are not doing that.
        # item = ScrapyDesignboomItem()

        # GRABS ALL LINKS TO THE 28 ARTICLES AT ONCE AND PUT THEM IN A LIST. THEN WE LOOP THROUGH THE LIST TO CLICK ON THOSE

        # for DEZEEN
        # links = response.xpath('/html/body/div[5]/div/main/ul/li/article/header/h3/a/@href').extract()

        links = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/h3[1]/a/@href').extract()
        # for i, href in enumerate(response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/h3[1]/a/@href')):
            # tags_xpath = '//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/div[2]/ul/li/a/text()'
            #
            # # // *[ @ id = "dboom-badge"] / div[1] / div / section / div[1] / div[10] / div / article[4] / div[2]
            # tags = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/div[2]/ul/li/a/text()').getall()
            # print("TAGS: ", tags)
            # if 'video' not in tags:


        # below is an example of trying to get the image shown on the main page
        # this is the valid one for small intro image
        # main_img = response.xpath('/html/body/div[1]/div[1]/div/section/div[1]/div/div/article/div[1]/a/img[2]/@data-lazy-src').extract()
        # item['image_main_url'] = main_img

        # for each link, calls the main function that does the data extraction
        for i, link in enumerate(links):
            print("parsing article: ", link)
            # yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True, meta={'db_item':item}) # ---> TO PASS ITEMS TO OTHER FUNCTIONS
            yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True)
            print('yielded')


        if self.counter == 0:

            # DEZEEN
            #/ html / body / div[5] / div / main / div[2] / aside / ol / li[7] / a
            next_pageurl = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a/@href')
        else:
            # DEZEEN
            #/ html / body / div[5] / div / main / div[2] / aside / ol / li[8] / a
            next_pageurl = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a[2]/@href')
        print("NEXT PAGE: ", next_pageurl)
        self.counter = 1

        # below a recursive function that calls itself to move to the next page if there is such next button.
        if next_pageurl:
            next_page = next_pageurl.extract_first()
            print("NEXT PAGE: ", next_page)

            yield scrapy.Request(next_page, callback=self.parse, dont_filter=True)

        # ALTERNATIVE BELOW --->
        # next_page = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a/@href').get()
        # if next_page is not None:
        #     yield response.follow(next_page, callback=self.parse)
        # <---------------------

    def parse_art_entry(self, response):
        """
        FOR EACH ARTICLE IN THE MAIN WEBSITE GETS THE DATA WE WANT. IMAGE + DESCRIPTION + TITLE
        :param response:
        :return:
        """

        # item = response.meta.get('db_item') <----- THIS IS IN CASE WE NEEDED TO USE THE ITEM IN THE MAIN FUNCTION

        # get the information of image and description
        def extract_img_with_xpath():
            # possible xpaths for images (I encountered that using only one wasn't working
            im_xpath = '//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/p/img/@data-src'
            im_xpath2 = '//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/img/p/@data-lazy-src'
            img = response.xpath(im_xpath).getall()
            if img == []:
                img = response.xpath(im_xpath2).getall()
            return img

        def extract_text_with_xpath(query):
            return response.xpath(query).getall()


        # These three lines below extract the data we want
        image = extract_img_with_xpath()
        desc = extract_text_with_xpath('//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/p')
        title = response.xpath('//*[@id="dboom-badge"]/header/div[4]/div[2]/div/h1/text()').get()

        # Puts the information the ITEM: Dictionary of the 3 fields we are interested in.
        yield ScrapyDesignboomItem(image_urls=image, text=desc, title=title)


        # item['text'] = desc
        # item['title'] = title
        # item['image_extra_urls'] = image
        #
        # yield item
        # yield field calls the above function for each of the queries



        # yield {
        #     'image': extract_img_with_xpath(),
        #     'text': extract_text_with_xpath('//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/p'),
        # }




# # TEXT
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[5]/p[11] # BANKSY
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[3]/p[7]  # THE OTHER
#
# # IMAGE
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[5]/p[6]/img # BANKSY
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[3]/p[10]/img # THE OTHER
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[3]/p[7]/img
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[3]/p[15]/img
# //*[@id="dboom-badge"]/div/div/section/div/div[1]/div[3]/p[32]/img
