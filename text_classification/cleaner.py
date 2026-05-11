from bs4 import BeautifulSoup


class Cleaner():
    def __init__(self):
        pass

    def put_line_breaks(self, text):
        # Insert newline after closing paragraph tags so block structure survives stripping
        return text.replace("</p>", "</p>\n")

    def remove_html_tags(self, text):
        # html.parser is built-in (no lxml dependency required)
        clean_text = BeautifulSoup(text, "html.parser").text
        return clean_text

    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        text = text.strip()
        return text
