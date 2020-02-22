# -*- coding: utf-8 -*-
import scrapy
from scrapy_designboom.items import ScrapyDesignboomItem
import pdb

class Designboom(scrapy.Spider):
    name = 'designboom_crawler'
    allowed_domains = ['designboom.com/art']
    start_urls = ['http://designboom.com/art/']

    def __init__(self):
        self.counter = 0

    def parse(self, response):
        if response.meta['depth'] > 270:
            print('Loop?')

        # item = ScrapyDesignboomItem()
        #follow links to each art entry
        links = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/h3[1]/a/@href').extract()
        # for i, href in enumerate(response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/h3[1]/a/@href')):
            # tags_xpath = '//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/div[2]/ul/li/a/text()'
            #
            # # // *[ @ id = "dboom-badge"] / div[1] / div / section / div[1] / div[10] / div / article[4] / div[2]
            # tags = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div/div/article/div[2]/ul/li/a/text()').getall()
            # print("TAGS: ", tags)
            # if 'video' not in tags:


        # this is the valid one for small intro image
        # main_img = response.xpath('/html/body/div[1]/div[1]/div/section/div[1]/div/div/article/div[1]/a/img[2]/@data-lazy-src').extract()
        # item['image_main_url'] = main_img

        for i, link in enumerate(links):
            print("parsing article: ", link)
            # yield response.follow(href, self.parse_art_entry)
            # yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True, meta={'db_item':item}) # ---> TO PASS ITEMS TO OTHER FUNCTIONS
            yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True)
            print('yielded')

        # follow pagination links
        # for pag in response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a/@href').extract():
        #     print("PAGINATION: ", pag)
        #     if pag != None:
        #         # yield response.follow(pag, callback=self.parse)
        #         yield scrapy.Request(pag, callback=self.parse, dont_filter=True)

        if self.counter == 0:
            next_pageurl = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a/@href')
        else:
            next_pageurl = response.xpath('//*[@id="dboom-badge"]/div[1]/div/section/div[1]/div[12]/a[2]/@href')
        print("NEXT PAGE: ", next_pageurl)
        self.counter = 1
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
        FOR EACH ARTICLE IN THE MAIN WEBSITE CALL THIS FUNCTION
        :param response:
        :return:
        """

        # item = response.meta.get('db_item')
        # get the information of image and description
        def extract_img_with_xpath():
            # possible xpaths for images
            im_xpath = '//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/p/img/@data-src'
            im_xpath2 = '//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/img/p/@data-lazy-src'
            img = response.xpath(im_xpath).getall()
            if img == []:
                img = response.xpath(im_xpath2).getall()
            return img

        def extract_text_with_xpath(query):
            return response.xpath(query).getall()


        image = extract_img_with_xpath()

        desc = extract_text_with_xpath('//*[@id="dboom-badge"]/div/div/section/div/div[1]/div/p')
        title = response.xpath('//*[@id="dboom-badge"]/header/div[4]/div[2]/div/h1/text()').get()

        # item['text'] = desc
        # item['title'] = title
        # item['image_extra_urls'] = image
        #
        # yield item
        # yield field calls the above function for each of the queries
        yield ScrapyDesignboomItem(image_urls=image, text=desc, title=title)

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
