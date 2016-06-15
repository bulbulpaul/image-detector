# coding:utf-8

import os
import re
import urllib2
from bs4 import BeautifulSoup
import requests


class BingHarvester():
    def get_soup(self, url):
        return BeautifulSoup(requests.get(url).text, 'html.parser')

    def execute(self, query, image_type):
        images = self.get_urls(query)

        for img in images:
            raw_img = urllib2.urlopen(img).read()
            cntr = len([i for i in os.listdir("src/resource/images") if image_type in i]) + 1
            f = open("src/resource/images/" + image_type + "_" + str(cntr) + '.jpg', 'wb')
            f.write(raw_img)
            f.close()

    def get_urls(self, query):
        url = "http://www.bing.com/images/search?q=" + query + \
              "&gft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

        soup = self.get_soup(url)
        images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]
        return images


if __name__ == '__main__':
    harvester = BingHarvester()
    harvester.execute('雨写真', 'rain')
    harvester.execute('豪雨', 'rain')
    harvester.execute('雷雨', 'rain')
    harvester.execute('晴天', 'sunny')
    harvester.execute('晴れた', 'sunny')
    harvester.execute('晴れ', 'sunny')

