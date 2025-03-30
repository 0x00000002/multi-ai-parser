from src.ai.enhanced_ai import AI
from src.ai.ai_config import Quality, Model
from src.ai.model_selector import UseCase
from src.logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():

    logging.getLogger().handlers = []    
    simple_logger = LoggerFactory.get_logger("logger", LoggingLevel.INFO, LogFormat.SIMPLE)
    
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