# -*- coding: utf-8 -*-
import scrapy
from dezeen.items import Dezeen_Item
from scrapy import signals
from scrapy.shell import inspect_response
import re


class DezeenCrawlerSpider(scrapy.Spider):
    # handle_httpstatus_list = [404, 301]
    name = 'dezeen_crawler'


    # ARCHITECTURE 789
    # allowed_domains = ['https://www.dezeen.com/architecture/']
    # start_urls = ['https://www.dezeen.com/architecture/']


    # PRODUCT DESIGN 68

    # allowed_domains = ['https://www.dezeen.com/tag/product-design/']
    # start_urls = ['https://www.dezeen.com/tag/product-design/']

    # FASHION 54
    # allowed_domains = ['https://www.dezeen.com/tag/fashion-tag/']
    # start_urls = ['https://www.dezeen.com/tag/fashion-tag/']

    # TECHNOLOGY 78
    # allowed_domains = ['https://www.dezeen.com/tag/technology-tag/']
    # start_urls = ['https://www.dezeen.com/tag/technology-tag/']

    # FURNITURE 133
    # allowed_domains = ['https://www.dezeen.com/tag/furniture/']
    # start_urls = ['https://www.dezeen.com/tag/furniture/']

    # ART
    # allowed_domains = ['https://www.dezeen.com/tag/art/']
    # start_urls = ['https://www.dezeen.com/tag/art/']

    # DESIGN
    allowed_domains = ['https://www.dezeen.com/design/']
    start_urls = ['https://www.dezeen.com/design/']

    def __init__(self):
        self.counter = 0
        self.failed_urls = []

    def parse(self, response):

        if response.status == 404:
            self.crawler.stats.inc_value('failed_url_count')
            self.failed_urls.append(response.url)

        if response.status == 301:
            self.crawler.stats.inc_value('failed_url_count_301')
            self.failed_urls.append(response.url)


        # this is to help debugging. I manually looked which page number is the last one.
        # if response.meta['depth'] > 69:
        #     print('Loop?')

        # NOT USING THIS, BUT WE CAN INTIATE THE ITEM (the main structure where we store the data) HERE, IF WE NEED TO
        # PASS SOME INFORMATION FROM THE MAIN PAGE.
        # WE ARE NOT USING THIS BECAUSE ANY INFORMATION COMES FROM CLICKING ON EACH SPECIFIC ENTRY OF THE WEBSITE.
        # Designboom has 28 entries per page. We individually entry in each of them. Say that you wanted to get the image
        # it shows in the main page, we would need to put the image in the ITEM here. But we are not doing that.
        # item = ScrapyDesignboomItem()

        # GRABS ALL LINKS TO THE 28 ARTICLES AT ONCE AND PUT THEM IN A LIST. THEN WE LOOP THROUGH THE LIST TO CLICK ON THOSE

        links = response.xpath('/html/body/div[4]/div/main/ul/li/article/header/h3/a/@href').getall()
        # '/html/body/div[4]/div/main/aside/h2'
        # '/html/body/div[4]/div/main/aside/h2'
        # '/html/body/div[4]/div/main/aside/h2'
        # '/html/body/div[4]/div/main/aside/ol/li[7]/a'
        # '/html/body/div[4]/div/main/aside/ol/li[6]/a'
        # inspect_response(response, self)
        # for each link, calls the main function that does the data extraction
        for i, link in enumerate(links):
            print("parsing article: ", link)
            # yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True, meta={'db_item':item}) # ---> TO PASS ITEMS TO OTHER FUNCTIONS
            yield scrapy.Request(link, callback=self.parse_art_entry, dont_filter=True)
            print('yielded')


        if self.counter == 0:
            #nspect_response(response, self)
            # next_pageurl = response.xpath('/html/body/div[4]/div/header/div[2]/aside/ol/li[7]/a/@href').extract_first()
            next_pageurl = response.xpath('/html/body/div[4]/div/main/aside/ol/li[7]/a/@href').extract_first()

        else:
            current_url = response.url
            current_page = re.findall('\d+', current_url)
            current_page = int(current_page[0])

            # CHECK IN
            # if current_page == 788:
            #     inspect_response(response, self)

            for i in range(15):
                title_path = '/html/body/div[4]/div/main/aside/ol/li[{}]/a/@title'.format(i)
                title = response.xpath(title_path).extract_first()
                if title == str(current_page + 1):
                    path = '/html/body/div[4]/div/main/aside/ol/li[{}]/a/@href'.format(i)
                    path2 = '/html/body/div[4]/div/header/div[2]/aside/ol/li[{}]/a/@href'.format(i)
                    next_pageurl = response.xpath(path).extract_first()
                    if next_pageurl: break
                    else:
                        next_pageurl = response.xpath(path2).extract_first()
                        if next_pageurl: break
        self.counter = 1
        # inspect_response(response, self)
        # below a recursive function that calls itself to move to the next page if there is such next button.
        if next_pageurl != None:
            # next_page = next_pageurl
            print("NEXT PAGE: ", next_pageurl)
            yield scrapy.Request(next_pageurl, callback=self.parse, dont_filter=True)


    def parse_art_entry(self, response):
        """
        FOR EACH ARTICLE IN THE MAIN WEBSITE GETS THE DATA WE WANT. IMAGE + DESCRIPTION + TITLE
        :param response:
        :return:
        """
        # item = response.meta.get('db_item') <----- THIS IS IN CASE WE NEEDED TO USE THE ITEM IN THE MAIN FUNCTION

        # get the information of image and description
        def extract_img_with_xpath(query):
            img = response.xpath(query).getall()
            return img

        def extract_text_with_xpath(query1, query2):
            main_part = response.xpath(query1).getall()
            hyperlinks = response.xpath(query2).getall()
            return main_part + hyperlinks


        # These three lines below extract the data we want
        image_path = '/html/body/div[4]/div/main/article/section/figure/img/@data-src'
        main_description = '/html/body/div[4]/div/main/article/section/p/text()'
        hyperlinks = '/html/body/div[4]/div/main/article/section/p/a/text()'

        image = extract_img_with_xpath(image_path)
        desc = extract_text_with_xpath(main_description, hyperlinks)

        title = response.xpath('/html/body/div[4]/div/main/article/header/h1/a/text()').get()

        # Puts the information the ITEM: Dictionary of the 3 fields we are interested in.
        yield Dezeen_Item(image_urls=image, text=desc, title=title)

    # SIGNALS

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(DezeenCrawlerSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.handle_spider_closed, signals.spider_closed)
        return spider


    def handle_spider_closed(self, reason):
        self.crawler.stats.set_value('failed_urls', ', '.join(self.failed_urls))

    def process_exception(self, response, exception, spider):
        ex_class = "%s.%s" % (exception.__class__.__module__, exception.__class__.__name__)
        self.crawler.stats.inc_value('downloader/exception_count', spider=spider)
        self.crawler.stats.inc_value('downloader/exception_type_count/%s' % ex_class, spider=spider)

