import scrapy
from bs4 import BeautifulSoup


class OnePieceSpider(scrapy.Spider):
    name = 'onepiecespider'

    # Category pages for each ability type — standard MediaWiki, always available
    start_urls = [
        'https://onepiece.fandom.com/wiki/Category:Devil_Fruits',
        'https://onepiece.fandom.com/wiki/Category:Haki',
        'https://onepiece.fandom.com/wiki/Category:Fighting_Styles',
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'FEEDS': {
            'data/abilities.jsonl': {'format': 'jsonlines', 'overwrite': True}
        },
    }

    def parse(self, response):
        # Extract links to individual ability pages from the category listing
        for href in response.css('div.category-page__members a.category-page__member-link::attr(href)').getall():
            yield response.follow(href, callback=self.parse_ability)

        # Follow "next page" pagination
        next_page = response.css('a.category-page__pagination-next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_ability(self, response):
        ability_name = response.css('span.mw-page-title-main::text').get('').strip()

        div_selector = response.css('div.mw-parser-output')
        if not div_selector:
            return

        soup = BeautifulSoup(div_selector.get(), 'html.parser').find('div')

        ability_type = ''
        if soup.find('aside'):
            aside = soup.find('aside')
            for cell in aside.find_all('div', {'class': 'pi-data'}):
                label = cell.find('h3') or cell.find('div', {'class': 'pi-data-label'})
                if label:
                    cell_name = label.text.strip()
                    # One Piece wiki uses "Type" for devil fruit type, "Classification" for others
                    if cell_name in ('Type', 'Classification'):
                        value = cell.find('div', {'class': 'pi-data-value'})
                        ability_type = value.text.strip() if value else cell.find('div').text.strip()

            soup.find('aside').decompose()

        ability_description = soup.text.strip()
        ability_description = ability_description.split('Trivia')[0].strip()
        ability_description = ability_description.split('References')[0].strip()

        if ability_name:
            yield dict(
                ability_name=ability_name,
                ability_type=ability_type,
                ability_description=ability_description,
            )
