from . import tools


def make_result(
    customer_id,
    order_id,
    agent,
    approved,
    reason,
    amount_cents=None,
    refund_id=None,
    reply=None,
    error=None,
):
    result = {
        "approved": approved,
        "reason": reason,
        "customer_id": customer_id,
        "order_id": order_id,
        "amount_cents": amount_cents,
        "refund_id": refund_id,
        "agent": agent,
    }
    if reply is not None:
        result["reply"] = reply
    if error is not None:
        result["error"] = error
    return result


def run_deterministic(customer_id, order_id):
    customer = tools.get_customer(customer_id)
    if "error" in customer:
        return make_result(customer_id, order_id, "deterministic", False, customer["error"])

    order = tools.get_order(order_id)
    if "error" in order:
        return make_result(customer_id, order_id, "deterministic", False, order["error"])

    if order["customer_id"] != customer_id:
        return make_result(customer_id, order_id, "deterministic", False, "not_order_owner")

    if order["status"] != "cancelled":
        return make_result(
            customer_id, order_id, "deterministic", False, "order_not_cancelled"
        )

    existing = tools.find_refund(order_id)
    if existing:
        return make_result(
            customer_id,
            order_id,
            "deterministic",
            False,
            "already_refunded",
            amount_cents=existing["amount_cents"],
            refund_id=existing["refund_id"],
        )

    refund = tools.record_refund(order_id)
    if "error" in refund:
        return make_result(customer_id, order_id, "deterministic", False, refund["error"])

    return make_result(
        customer_id,
        order_id,
        "deterministic",
        True,
        "refund_approved",
        amount_cents=refund["amount_cents"],
        refund_id=refund["refund_id"],
    )


def run_refund_agent(customer_id, order_id, agent="deterministic"):
    customer_id = (customer_id or "").strip()
    order_id = (order_id or "").strip()

    if not customer_id or not order_id:
        return make_result(
            customer_id,
            order_id,
            agent,
            False,
            "invalid_request",
            error="customer_id and order_id are required",
        )

    if agent == "deterministic":
        return run_deterministic(customer_id, order_id)

    if agent == "llm":
        from .llm import run_llm_agent

        return run_llm_agent(customer_id, order_id)

    return make_result(
        customer_id,
        order_id,
        agent,
        False,
        "unknown_agent_mode",
        error=f"unknown agent mode: {agent}",
    )
