from src.ai.AI import AI
from src.ai.AIConfig import Quality, Speed, Model
from src.ai.ModelSelector import ModelSelector, UseCase
from src.Website import Website
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():

    logging.getLogger().handlers = []    
    simple_logger = LoggerFactory.get_logger("translator", LoggingLevel.INFO, LogFormat.SIMPLE)
    
    coding_ai = AI.for_use_case(use_case=UseCase.CODING, quality=Quality.MEDIUM, speed=Speed.STANDARD)
    print(coding_ai.model)
    coding_ai.logger = simple_logger
    local_res = coding_ai.request("Find a ticket price for a return ticket to Paris.")
    print(local_res)

if __name__ == "__main__":
    main()    