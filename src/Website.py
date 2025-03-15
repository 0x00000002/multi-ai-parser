#!/usr/bin/env python
# coding: utf-8

# In[1]:
import requests
from IPython.display import Markdown, display
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# In[2]:
class Website:
    def __init__(self, url, translateTo=None):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        self.fetch()
        self.use_soup()

    def fetch(self):
        logger.info(f"Fetching {self.url}...")        
        response = requests.get(self.url, headers=headers)
        self.body = response.content

    def use_soup(self):
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        self.images = [img.get('src') for img in soup.find_all('img')]
        if soup.body:
            for irrelevant in soup.body(["script", "style", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title} \n\
            Webpage Contents:\n{self.text} \n\
            Images:\n{self.images} \n\
            Links:\n{self.links} \n\n"




# In[3]:
ws = Website("https://www.google.com")
print(ws.get_contents())
# %%
