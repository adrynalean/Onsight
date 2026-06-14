import re
from urllib.parse import urlencode

import scrapy
from bs4 import BeautifulSoup


class OnePieceSpider(scrapy.Spider):
    """Crawl One Piece ability data via the MediaWiki API and emit
    (name, type, description) rows.

    Why the API and not HTML scraping: onepiece.fandom.com sits behind
    Cloudflare and returns HTTP 403 to Scrapy on every ``/wiki/`` page (even
    with full browser headers), but the ``api.php`` endpoint responds 200. The
    API also gives clean JSON, avoids fragile CSS selectors, and exposes the
    category tree so we can recurse.

    Each seed is tagged with the simplified ability label it should produce.
    Category seeds are crawled recursively because the real Devil Fruit pages
    live in subcategories (Paramecia / Logia / Zoan), not directly under
    Category:Devil_Fruits. Category:Haki is empty on the wiki, so Haki is seeded
    from its explicit article pages instead.
    """

    name = 'onepiecespider'
    api_url = 'https://onepiece.fandom.com/api.php'

    # category title -> simplified ability label
    category_seeds = {
        'Category:Devil Fruits': 'Devil Fruit',
        'Category:Fighting Styles': 'Physical Technique',
    }
    # Category:Haki has zero members; crawl the real Haki articles directly.
    haki_pages = [
        'Haki',
        'Haki/Armament Haki',
        'Haki/Observation Haki',
        'Haki/Supreme King Haki',
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 0.5,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 4,
        'USER_AGENT': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'FEEDS': {
            'data/abilities.jsonl': {'format': 'jsonlines', 'overwrite': True}
        },
    }

    # ----- request builders -------------------------------------------------

    def _api(self, **params):
        return f'{self.api_url}?{urlencode(params)}'

    def _category_request(self, category, ability_type, extra=None):
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmtype': 'page|subcat',
            'cmlimit': '500',
        }
        if extra:
            params.update(extra)
        return scrapy.Request(
            self._api(**params),
            callback=self.parse_category,
            cb_kwargs={'ability_type': ability_type, 'category': category},
        )

    def _page_request(self, ability_type, *, pageid=None, page=None):
        params = {'action': 'parse', 'format': 'json', 'prop': 'text'}
        if pageid is not None:
            params['pageid'] = str(pageid)
        else:
            params['page'] = page
        return scrapy.Request(
            self._api(**params),
            callback=self.parse_ability,
            cb_kwargs={'ability_type': ability_type},
        )

    def _seed_requests(self):
        for category, ability_type in self.category_seeds.items():
            yield self._category_request(category, ability_type)
        for page in self.haki_pages:
            yield self._page_request('Haki', page=page)

    # Scrapy <= 2.12 entry point.
    def start_requests(self):
        yield from self._seed_requests()

    # Scrapy >= 2.13 entry point (start_requests was removed in 2.13). Defining
    # both keeps the spider working whether requirements pins 2.12 or newer.
    async def start(self):
        for request in self._seed_requests():
            yield request

    # ----- parsing ----------------------------------------------------------

    def parse_category(self, response, ability_type, category):
        data = response.json()
        members = data.get('query', {}).get('categorymembers', [])

        for member in members:
            if member.get('ns') == 14:
                # Recurse every subcategory — including Non-Canon groupings, which
                # hold ~50 legitimate movie/game Devil Fruits and Fighting Styles.
                # Scrapy's dupefilter drops pages reached via multiple paths.
                yield self._category_request(member['title'], ability_type)
            else:
                yield self._page_request(ability_type, pageid=member['pageid'])

        # Paginate the current category.
        if 'continue' in data:
            yield self._category_request(category, ability_type, extra=data['continue'])

    def parse_ability(self, response, ability_type):
        data = response.json()
        parse = data.get('parse')
        if not parse:
            return

        ability_name = (parse.get('title') or '').strip()
        if not ability_name:
            return

        html = parse.get('text', {}).get('*', '')
        soup = BeautifulSoup(html, 'html.parser')
        root = soup.find('div', class_='mw-parser-output') or soup

        # Strip non-prose chrome: infobox, reference superscripts, tables,
        # table-of-contents, nav boxes, edit links, figures.
        for tag in root.find_all(['aside', 'sup', 'table', 'style', 'script', 'nav', 'figure']):
            tag.decompose()
        for node in root.select('.toc, #toc, .navbox, .noprint, .mw-editsection, .reference'):
            node.decompose()

        # Keep paragraph boundaries so Haki pages can be split into chunks.
        text = root.get_text('\n')
        for marker in ('\nTrivia', '\nReferences', '\nSite Navigation', '\nExternal Links'):
            text = text.split(marker)[0]
        text = re.sub(r'\[\s*\d+\s*\]', '', text)          # leftover [1] ref marks
        text = re.sub(r'(?im)^.*has been featured.*$', '', text)  # featured-article notice
        text = re.sub(r'\n{2,}', '\n', text).strip()

        if len(text.split()) < 12:
            return

        # Haki comes from only a handful of pages; split it into chunks so the
        # class isn't starved relative to Devil Fruits / Fighting Styles.
        if ability_type == 'Haki':
            for index, chunk in enumerate(self._chunk(text), start=1):
                yield dict(
                    ability_name=f'{ability_name} - section {index:02d}',
                    ability_type=ability_type,
                    ability_description=chunk,
                )
        else:
            yield dict(
                ability_name=ability_name,
                ability_type=ability_type,
                ability_description=re.sub(r'\s+', ' ', text).strip(),
            )

    @staticmethod
    def _chunk(text, min_words=25, max_words=120):
        """Group paragraphs into ~max_words chunks (paragraph-aligned)."""
        chunks = []
        current = []
        count = 0
        for paragraph in re.split(r'\n+', text):
            paragraph = paragraph.strip()
            words = paragraph.split()
            if len(words) < 8:
                continue
            if current and count + len(words) > max_words:
                if count >= min_words:
                    chunks.append(' '.join(current))
                current = []
                count = 0
            current.append(paragraph)
            count += len(words)
        if count >= min_words:
            chunks.append(' '.join(current))
        return chunks or [re.sub(r'\s+', ' ', text).strip()]
