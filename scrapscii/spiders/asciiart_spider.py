import scrapy

import scrapscii.unicode

# TARGETS ######################################################################


TARGET_DICT = {
    'animals': ['aardvarks', 'amoeba', 'bats', 'bears', 'beavers', 'birds-land', 'birds-water', 'bisons', 'camels', 'cats', 'cows', 'deer', 'dogs', 'dolphins', 'elephants', 'fish', 'frogs', 'insects/ants', 'insects/bees', 'insects/beetles', 'insects/butterflies', 'insects/caterpillars', 'insects/cockroaches', 'insects/other', 'insects/snails', 'insects/worms', 'horses', 'marsupials', 'monkeys', 'moos', 'other-land', 'other-water', 'rabbits', 'reptiles/alligators', 'reptiles/dinosaurs', 'reptiles/lizards', 'reptiles/snakes', 'rhinoceros', 'rodents/mice', 'rodents/other', 'scorpions', 'spiders', 'wolves', ],
    'logos': ['amnesty-international', 'biohazards', 'caduceus', 'coca-cola', 'hello-kitty', 'jolly-roger', 'kool-aid', 'no-bs', 'no-smoking', 'other', 'peace', 'pillsbury-doughboy', 'playboy', 'recycle', 'television'],}

# ASCII ARCHIVE ################################################################

class AsciiArtSpider(scrapy.Spider):
    name = 'asciiart'

    # META #####################################################################

    urls = [
        f'https://www.asciiart.eu/{__c}/{__i}'
        for __c, __l in TARGET_DICT.items()
        for __i in __l]

    # SCRAPING #################################################################

    def start_requests(self):
        for __u in self.urls:
            yield scrapy.Request(url=__u, callback=self.parse)

    # PARSING ##################################################################

    def parse(self, response):
        for __item in response.css('div.asciiarts > div'):
            # parse
            __all = __item.css('::text').getall()
            if __all:
                # capture
                __caption = ''.join(__all[:-1])
                __content = __all[-1]
                # format
                yield {
                    'caption': __caption,
                    'content': __content,
                    'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
                    'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),}
