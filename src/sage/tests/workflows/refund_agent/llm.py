import os

from langchain.agents import create_agent
from langchain.tools import tool

from . import tools

SYSTEM_PROMPT = """You are a refund approval agent.

Follow these rules for every request:
1. The customer must exist.
2. The order must exist.
3. The order must belong to the requesting customer. Compare the order's customer_id with the supplied customer id.
4. The order status must be exactly "cancelled".
5. Never refund an order that already has a refund. Check get_refund before refunding.

Use get_customer, get_order, get_refund, and record_refund to complete the request.
The refund amount is read from the order by record_refund, so you do not provide it.
If a check fails, do not call record_refund and explain the failure in one sentence.
Do not claim a refund happened unless you called record_refund.
"""


@tool
def get_customer(customer_id: str) -> dict:
    """Look up a customer by id. Returns the customer record or an error."""
    return tools.get_customer(customer_id)


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id. Returns the order record or an error."""
    return tools.get_order(order_id)


@tool
def get_refund(order_id: str) -> dict:
    """Look up an existing refund for an order. Returns the refund or an error."""
    refund = tools.find_refund(order_id)
    return refund if refund else {"error": "refund_not_found"}


@tool
def record_refund(order_id: str) -> dict:
    """Record a refund for an order. The amount is read from the order record."""
    return tools.record_refund(order_id)


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return " ".join(part for part in parts if part)
    return str(content)


def run_llm_agent(customer_id, order_id):
    from .agent import make_result

    if not os.environ.get("OPENROUTER_API_KEY"):
        return make_result(
            customer_id,
            order_id,
            "llm",
            False,
            "llm_config_error",
            error="OPENROUTER_API_KEY is not set",
        )

    model = os.environ.get("REFUND_AGENT_MODEL", "google/gemini-3.5-flash-lite")
    before = {refund["refund_id"] for refund in tools.list_refunds()}

    try:
        agent = create_agent(
            model=f"openrouter:{model}",
            tools=[get_customer, get_order, get_refund, record_refund],
            system_prompt=SYSTEM_PROMPT,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Review this refund request.\n"
                            f"customer_id: {customer_id}\n"
                            f"order_id: {order_id}"
                        ),
                    }
                ]
            },
            config={"recursion_limit": 12},
        )
    except Exception as exc:
        return make_result(
            customer_id,
            order_id,
            "llm",
            False,
            "llm_error",
            error=str(exc),
        )

    messages = result.get("messages", [])
    reply = _as_text(messages[-1].content) if messages else ""

    new_refunds = [
        refund
        for refund in tools.list_refunds()
        if refund["refund_id"] not in before and refund["order_id"] == order_id
    ]
    if new_refunds:
        refund = new_refunds[0]
        return make_result(
            customer_id,
            order_id,
            "llm",
            True,
            "refund_approved",
            amount_cents=refund["amount_cents"],
            refund_id=refund["refund_id"],
            reply=reply,
        )

    existing = tools.find_refund(order_id)
    if existing:
        return make_result(
            customer_id,
            order_id,
            "llm",
            False,
            "already_refunded",
            amount_cents=existing["amount_cents"],
            refund_id=existing["refund_id"],
            reply=reply,
        )

    return make_result(
        customer_id,
        order_id,
        "llm",
        False,
        "no_refund_recorded",
        reply=reply,
    )
