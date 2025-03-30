#!/usr/bin/env python
# coding: utf-8

import requests
from IPython.display import Markdown, display
import logging
from bs4 import BeautifulSoup
from src.ai import AI
from src.ai.ai_config import Quality, Speed
from src.prompter import Prompter
from src.ai.model_selector import ModelSelector, UseCase
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Callable
import json
from src.logger import Logger, NullLogger
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

@dataclass
class Website:
    url: str 
    title: str 
    text: str 
    images: list[str] 
    links: list[str] 
    url_translated: str 
    title_translated: str 
    text_translated: str 
    links_json: list[str] 
    images_json: list[str] 
    translated_links_json: list[str] 
    # Optional AI factory function for dependency injection
    ai_factory: Callable[..., Any] = None

    class Parsing_Error(Exception):
        """Custom exception for parsing-related errors."""
        pass

    
    def __init__(self, url: str, ai_factory: Optional[Callable[..., Any]] = None, logger: Optional[Logger] = None):
        """
        Create this Website object from the given url using the BeautifulSoup library
        
        Args:
            url: The URL to fetch
            ai_factory: Optional factory function to create AI instances
                        (for dependency injection)
        """
        self.url = url
        self.ai_factory = ai_factory or self._default_ai_factory
        self.fetch()
        self.use_soup()
        self._logger = logger if logger is not None else NullLogger()

    def _default_ai_factory(self, model_params: Dict[str, Any], system_prompt: str) -> AI:
        """Default AI factory method if none is provided."""
        return AI(model_params, system_prompt=system_prompt)

    def fetch(self):
        self._logger.info(f"Fetching {self.url}...")        
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
        self.links_json = json.dumps({"links": self.links}, indent=2)
        self.images_json = json.dumps({"images": self.images}, indent=2)

    def get_contents(self):
        return f"Webpage Title:\n{self.title} \n\
            Webpage Contents:\n{self.text} \n\
            Images:\n{self.images} \n\
            Links:\n{self.links} \n\n"

    def translateTo(self, translateTo: str, quality: Quality = Quality.MEDIUM, speed: Speed = Speed.STANDARD, use_local: bool = False):
        """
        Translate website content to the specified language.
        
        Args:
            translateTo: Target language for translation
            quality: Translation quality level from AIConfig.Quality
            speed: Speed preference from AIConfig.Speed
            use_local: Whether to use local models (default: False)
        """
        # Get model parameters using shared ModelSelector
        model_params = ModelSelector.get_model_params(
            use_case=UseCase.TRANSLATION,
            quality=quality,
            speed=speed,
            use_local=use_local
        )
        
        # Get the appropriate system prompt
        system_prompt = ModelSelector.get_system_prompt(UseCase.TRANSLATION)
        
        # Use the factory to create the translator
        translator = self.ai_factory(model_params, system_prompt)
        
        self._logger.info(f"Translating to {translateTo} using {translator.model.name}")
        
        # Translate the title
        self.title_translated = translator.request(
            Prompter.get_prompt_for_translation(translateTo, self.title)
        )
        self._logger.info(self.title_translated)
        
        # Translate the content (uncomment when ready)
        # self.text_translated = translator.request(
        #     Prompter.get_prompt_for_translation(translateTo, self.text)
        # )
        
        # Translate the links (uncomment when ready)
        # self.translated_links_json = translator.request(
        #     Prompter.get_prompt_for_translation_json(translateTo, self.links_json)
        # )


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
            raise self.Parsing_Error("Please parse_links() first")

        links = json.loads(self.parsed_links) 
        result = ""
        for link in links["links"]:
            result += Website(link["url"]).parse_text(Format.MARKDOWN, AIConfig.CHATGPT,"")
        self.with_text_from_links = result
        ai = AI(ai_config, Prompter.system_prompt)
        ai.request("analyse page text", Prompter.get_prompt_with(user_prompt, self.with_text_from_links))
        self.parsed_with_text_from_links = ai.results        