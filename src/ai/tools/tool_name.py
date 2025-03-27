from src.ai.tools.models import Function, Parameters, Property

def function_name(destination_city: str) -> str:
    return f"The price of a return ticket to {destination_city} is 1000 USD"

tool_name = Function(
    name="function_name",
    description="Tool description",
    parameters=Parameters(
        type="object",
        properties={"destination_city": Property(type="string", description="City name")},
        required=["destination_city"]
    )
)

