import scrapy

import scrapscii.unicode

# COPYPASTA ####################################################################

class TwitchQuotesSpider(scrapy.Spider):
    name = 'twitchquotes'

    # META #####################################################################

    urls = [
        f'https://www.twitchquotes.com/copypastas/ascii-art?page={__i}'
        for __i in range(1, 54)]

    # SCRAPING #################################################################

    def start_requests(self):
        for __u in self.urls:
            yield scrapy.Request(url=__u, callback=self.parse)

    # PARSING ##################################################################

    def parse(self, response):
        for __pasta in response.css('article.twitch-copypasta-card'):
            # parse
            __caption = __pasta.css('h3.-title-inner-parent::text').get()
            __content = __pasta.css('span.-main-text::text').get()
            __labels = __pasta.css('h4.tag-label::text').getall()
            # format
            if __caption and __content:
                yield {
                    'caption': __caption.strip(),
                    'content': __content,
                    'labels': ','.join(__l.strip().capitalize() for __l in set(__labels) if __l.strip()),
                    'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
                    'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),}
