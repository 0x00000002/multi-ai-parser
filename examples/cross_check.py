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
    
    coding_ai = AI.for_use_case(
        use_case=UseCase.CODING,
        quality=Quality.HIGH,
        use_local=True  # Use local models for coding
    )
    print(coding_ai.model)
    coding_ai.logger = simple_logger
    local_res = coding_ai.request("Explain the Ethereum's ERC7201 standard (not ERC721).")
    print(local_res)
    simple_logger.info("Checking with Claude...")
    claude_ai = AI(Model.CLAUDE_SONNET_3_7, system_prompt="Check this explanation of the Ethereum's ERC7201 standard (not ERC721).")
    claude_res = claude_ai.request("Check if the explanation is correct and complete the missing parts." + local_res)
    print(claude_res)
    

if __name__ == "__main__":
    main()    