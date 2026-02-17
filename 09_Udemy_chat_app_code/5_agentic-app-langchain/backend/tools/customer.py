import json
import os


def _data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_customers():
    path = os.path.join(_data_dir(), "customers.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_customers(limit: int | None = None) -> list:
    """Retrieve all customers from the json data based on limit.
    Fetch all customers.
    """
    customers = _load_customers()
    customers = sorted(customers, key=lambda c: c.get("joinedAt", ""), reverse=True)
    if limit is not None and limit > 0:
        return customers[:limit]
    return customers


def get_customer_by_id(id: str) -> dict | None:
    """Retrieve customer from the json data based on id.
    Fetch customer by id.
    """
    customers = _load_customers()
    for c in customers:
        if c.get("_id") == id:
            return c
    return None
