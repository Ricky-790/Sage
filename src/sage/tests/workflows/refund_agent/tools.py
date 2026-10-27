import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CUSTOMERS_PATH = DATA_DIR / "customers.json"
ORDERS_PATH = DATA_DIR / "orders.json"
REFUNDS_PATH = DATA_DIR / "refunds.json"


def _read(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def get_customer(customer_id):
    """Look up a customer by id."""
    for customer in _read(CUSTOMERS_PATH):
        if customer["customer_id"] == customer_id:
            return customer
    return {"error": "customer_not_found"}


def get_order(order_id):
    """Look up an order by id."""
    for order in _read(ORDERS_PATH):
        if order["order_id"] == order_id:
            return order
    return {"error": "order_not_found"}


def list_refunds():
    """Return every recorded refund."""
    return _read(REFUNDS_PATH)


def find_refund(order_id):
    """Return the refund for an order, if one exists."""
    for refund in list_refunds():
        if refund["order_id"] == order_id:
            return refund
    return None


def record_refund(order_id):
    """Record a refund using the amount stored on the order."""
    order = get_order(order_id)
    if "error" in order:
        return order

    refund = {
        "refund_id": f"R-{uuid.uuid4().hex[:8]}",
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "amount_cents": order["amount_cents"],
        "currency": order["currency"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    refunds = list_refunds()
    refunds.append(refund)
    _write(REFUNDS_PATH, refunds)
    return refund
