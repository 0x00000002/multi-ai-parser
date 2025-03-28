from src.ai.tools.models import Function, Parameters, Property

def get_ticket_price(destination_city: str) -> str:
    return f"The price of a return ticket to {destination_city} is 1000 USD"

ticket_price_tool = Function(
    name="get_ticket_price",
    description="Allows to get the price of a ticket to a destination city. Call this tool whenever user asks about the tickets prices to any city.",
    parameters=Parameters(
        type="object",
        properties={"destination_city": Property(type="string", description="City name")},
        required=["destination_city"]
    )
)

