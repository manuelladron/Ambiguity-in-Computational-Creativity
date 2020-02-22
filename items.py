# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapyDesignboomItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # image_urls = scrapy.Field()
    # images = scrapy.Field()

    image_urls = scrapy.Field()
    images = scrapy.Field()

    # image_main_url = scrapy.Field()
    # image_main = scrapy.Field()

    text = scrapy.Field()
    title = scrapy.Field()


# to use other naming! Not really necessary
# IMAGES_URLS_FIELD = 'image_urls'
# IMAGES_RESULT_FIELD = 'images'


