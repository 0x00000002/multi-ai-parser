#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython.display import Markdown, display
from enum import Enum
import os
import json
from pydantic import BaseModel, TypeAdapter
from typing import List, Dict, Any, Optional
import logging
import re
from src.Website import Website
from src.Prompter import Prompter



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# In[2]:

class Image(BaseModel):
    alt: str
    description: str
    url: str

class Link(BaseModel):
    category: str
    name: str
    url: str


class Parser:
    class Error(Exception):
        """Custom exception for parsing-related errors."""
        pass

    class ParsingResult(BaseModel):
        title: str
        text: str
        images: List[Image]
        links: List[Link]


    def __init__(self, url):
        """
        Fetch and parse the Website object from the given url
        """

        self.website = Website(url)
        self.results = self.ParsingResult(
            title=str(self.website.title),
            text="",  # Empty string instead of str type
            images=[],  # Empty list instead of list[Image]
            links=[]    # Empty list instead of list[Link]
        )
        self.translation = self.ParsingResult(
            title="",
            text="",
            images=[],
            links=[]
        )

    @staticmethod
    def extract_text(text, startPattern=None, endPattern=None):
        startPattern = "" if startPattern is None else re.escape(startPattern)
        endPattern = "$" if endPattern is None else re.escape(endPattern)        
        pattern = f"{startPattern}(.*?){endPattern}"

        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


    def translate(self, ai_model,to_language):
        ai = AI(ai_model, self.translate_prompt)
        ai.request("translate", self.get_prompt_for_translation(to_language, self.results.title))
        self.translation.title = ai.results
        ai.request("translate", self.get_prompt_for_translation(to_language, self.results.text))
        self.translation.text = ai.results
        for link in self.results.links:
            ai.request("translate", self.get_prompt_for_translation(to_language, link.name))
            self.translation.links.append(ai.results)
        for image in self.results.images:
            ai.request("translate", self.get_prompt_for_translation(to_language, image.description))
            self.translation.images.append(ai.results)


    # def format_links_for_analysis(links):
    #     """Format a list of links into a string suitable for AI analysis"""
    #     if isinstance(links, list):
    #         return "Please analyze these links:\n" + "\n".join(str(link) for link in links)
    #     return str(links)

    def parse_text(self, ai_config, user_prompt):
        if not self.website.text:
            raise Parser.Error("It seems that there is no text for this website")

        prompt = user_prompt
        prompt += f"You are looking at a website titled {self.website.title} with the following content:\n"
        prompt += self.website.text

        ai = AI(ai_config, Prompter.system_prompt)
        ai.request("analyse page text", prompt)
        self.results.text=ai.results


    def parse_links(self, ai_config):  
        prompt = Prompter.get_prompt_for_links_json("\n".join(str(link) for link in self.website.links))
        # print(prompt)
        ai = AI(ai_config, Prompter.system_prompt)
        ai.request("get links", prompt)
        self.parsed_links = ai.results.replace("```json", "").replace("```", "")
        stripped_results = ai.results.strip()

        parsed_data = json.loads(self.parsed_links)
        # print(parsed_data)
        # self.results.links = parse_obj_as(List[Link], parsed_data["links"])

    def parse_pages_from_links(self, ai_config, user_prompt):
        if not self.results.links:
            raise Parser.Error("Please parse_links() first")

        links = json.loads(self.parsed_links) 
        result = ""
        for link in links["links"]:
            result += Website(link["url"]).parse_text(Format.MARKDOWN, AIConfig.CHATGPT,"")
        self.with_text_from_links = result
        ai = AI(ai_config, Prompter.system_prompt)
        ai.request("analyse page text", Prompter.get_prompt_with(user_prompt, self.with_text_from_links))
        self.parsed_with_text_from_links = ai.results

    def print_report(self):
        display(Markdown(self.parsed_text)) if self.parsed_text else "Please parse_text() first"
        display(Markdown(self.parsed_links)) if self.parsed_links else "Please parse_links() first"

