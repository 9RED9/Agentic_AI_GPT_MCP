import json
import os


def _data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_orders():
    path = os.path.join(_data_dir(), "orders.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_customers():
    path = os.path.join(_data_dir(), "customers.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_orders(limit: int | None = None) -> list:
    """Retrieve all orders from the json data based on limit.
    Fetch all orders.
    """
    orders = _load_orders()
    orders = sorted(orders, key=lambda o: o.get("date", ""), reverse=True)
    if limit is not None and limit > 0:
        return orders[:limit]
    return orders


def get_orders_with_customer_details(limit: int | None = None) -> list:
    """Retrieve the latest orders along with customer details (name) with optional limit.
    Fetch all orders with customer details.
    """
    customers = {c["_id"]: c for c in _load_customers()}
    orders = _load_orders()
    out = []
    for o in orders:
        o = dict(o)
        cid = o.get("customer")
        o["customer"] = customers.get(cid, {}).get("name", cid) if isinstance(cid, str) else cid
        out.append(o)
    out.sort(key=lambda x: x.get("date", ""), reverse=True)
    if limit is not None and limit > 0:
        return out[:limit]
    return out


def get_order_by_id(id: str) -> dict | None:
    """Retrieve order from the json data based on id.
    Fetch order by id.
    """
    orders = _load_orders()
    for o in orders:
        if o.get("_id") == id:
            return o
    return None
