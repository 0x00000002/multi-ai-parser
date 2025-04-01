# Agentic-AI Project Guidelines

## Build & Test Commands
- Setup: `conda env create -f environment.yml` or `pip install -r requirements.txt`
- Install dev: `pip install -e .`
- Run all tests: `python -m unittest discover -s tests`
- Run specific module tests: `python -m unittest discover -s tests/tools`
- Run tools test suite: `python tests/tools/run_tools_tests.py`
- Run single test file: `python -m unittest tests/core/test_base_ai.py`
- Run specific test: `python -m unittest tests.core.test_base_ai.TestClassName.test_method_name`

## Code Style Guidelines
- **Imports**: Organize by stdlib → third-party → project modules
- **Type Hints**: Required for all function parameters and return values
- **Naming**:
  - Classes: PascalCase (e.g., `ConfigManager`)
  - Methods/Functions/Variables: snake_case
  - Interfaces: Suffix with "Interface" (e.g., `LoggerInterface`)
- **Error Handling**: Use custom exceptions from `exceptions.py`
- **Documentation**: Docstrings for all classes and methods
- **Architecture**: Follow modular design with clear separation of concerns
- **Formatting**: 4-space indentation, 120 character line limit

## Project Structure
- Core AI functionality in `src/core/`
- Provider implementations in `src/core/providers/`
- Tool functionality in `src/tools/`
- Configuration in `src/config/`