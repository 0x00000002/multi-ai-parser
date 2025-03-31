from src.core.tool_enabled_ai import AI
from src.config.models import Model
from src.config.config_manager import Quality, Speed
from src.core.model_selector import UseCase, ModelSelector
from src.utils.logger import LoggerFactory, LoggingLevel, LogFormat
from src.config.config_manager import ConfigManager
# Example usage with shared model selection
def main():

    
    simple_logger = LoggerFactory.create("logger", LoggingLevel.INFO, LogFormat.SIMPLE, use_real_logger=True)
    
    config_manager = ConfigManager(config_path="src/config/config.yml")
    # coding_model_config = config_manager.get_use_case_config(UseCase.CODING.name.lower())
    model_selector = ModelSelector(config_manager)
    model = model_selector.select_model(use_case=UseCase.CODING, quality=Quality.HIGH, speed=Speed.FAST)
    print(model)
    ai = AI(
        model=model,  # Use Model enum
        config_manager=config_manager,
        logger=simple_logger
    )
    
    local_res = ai.request("Explain the Ethereum's ERC7201 standard (not ERC721).")
    print(local_res)
    simple_logger.info("Checking with Claude...")
    claude_ai = AI(model=Model.CLAUDE_3_7_SONNET, system_prompt="Check this explanation of the Ethereum's ERC7201 standard (not ERC721).")
    claude_res = claude_ai.request("Check if the explanation is correct and complete the missing parts." + local_res)
    print(claude_res)
    

if __name__ == "__main__":
    main()    