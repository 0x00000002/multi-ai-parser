from src.content_types import Format

class Prompter:
    system_prompt = """
        You are an assistant who reads a text and helps to answer the questions from user.
        Don't guess the answer, if you don't know the answer, say that you don't know.
        Response in English, format your response in Markdown.
        """

    prompt_for_links = """
        Please analyze the provided below list of web links. \
        Remove non-relevant links, e.g. links to LinkedIn, or empty links, or JS popups. \
        Remove links to social media, subscription pages, legal information, \
        news, events, articles, blog posts, external websites, \
        different languages duplicates of the same page, and any other non-relevant links.
        Create the name for every link, and group all links by type.
        Here is the list of links:
        """

    class FormatError(Exception):
        """Custom exception for format-related errors."""
        pass

    @staticmethod
    def get_prompt_for_translation(to_language, text):
        prompt = f"Translate the following text to the {to_language} language."
        prompt += text
        return prompt

    @staticmethod
    def get_prompt_for_translation(translateTo:str, text:str):
        prompt = f"Translate the following text to the {translateTo} language.\n"
        prompt += text
        return prompt
    
    @staticmethod
    def get_prompt_for_translation_json(translateTo:str, stringified_json:str):
        prompt = f"Translate ONLY the values of the following JSON into {translateTo} language. Don't translate links/urls.\n"
        prompt += stringified_json
        prompt += "Respond in the same JSON format"
        return prompt

    @staticmethod
    def get_prompt_for_links_json(text):
        prompt = Prompter.prompt_for_links
        prompt += text
        prompt += "Respond in JSON format, as in this example:"
        prompt += """
            {
                "links": [
                    {"category": "Related websites", "name":"about page", "url": "https://full.url/goes/here/about"},
                    {"category": "Programs", "name": "Current Programs page", "url": "https://full.url/current-programs"}
                ]
            }
            don't include ```json or ``` in the response
            """
        return prompt

    @staticmethod
    def get_prompt_for_links_markdown(text):
        prompt = Prompter.prompt_for_links
        prompt += text
        prompt += "Respond in Markdown, as in this examle:"
        prompt += """
            ## Links
            ### Programs
            - [Program 10/20](https://another.full.url/program_10_20)
            ### Related websites
            - [Related website 1](https://another.full.url/related_website_1)
            - [Related website 2](https://another.full.url/related_website_2)
            """
        return prompt

    @staticmethod   
    def get_prompt_for_links(format, text):
        if format == Format.JSON:
            return Prompter.get_prompt_for_links_json(text)
        elif format == Format.MARKDOWN:
            return Prompter.get_prompt_for_links_markdown(text)
        else:
            raise Prompter.FormatError(f"Format is not supported: {format}")
