import re

class Parser:
    class Parsing_Error(Exception):
        """Custom exception for parsing-related errors."""
        pass

    @staticmethod
    def extract_text(text, startPattern=None, endPattern=None):
        try:
            startPattern = "" if startPattern is None else re.escape(startPattern)
            endPattern = "$" if endPattern is None else re.escape(endPattern)        
            pattern = f"{startPattern}(.*?){endPattern}"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        except Exception as e:
            raise Parser.Parsing_Error(f"Error extracting text: {str(e)}")


