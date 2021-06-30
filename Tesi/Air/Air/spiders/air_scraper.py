import scrapy


class AirSpider(scrapy.Spider):
    name = "AirSpider"

    def start_requests(self):
        urls = [
            'https://bollettino.appa.tn.it/aria/opendata/json/2021-02-24,2021-05-12/'
     ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'quotes-{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'Stazione': quote.css('span.text::text').get(),
                'Inquinante': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }