
from datetime import datetime, timezone
import hashlib
from langchain.tools import tool
import httpx

CRM_BASE_URL = "https://crm.tailorsin.com/tailorsin-api/api"
REQUEST_TIMEOUT = 15.0

# CRM type → internal segment name mapping
# active_client  = current order in progress
# client         = past orders but no active order
# lead           = registered but no orders placed
# (not found)    = new_user — needs to register
_TYPE_TO_SEGMENT = {
    "active_client": "active_client",
    "client": "client",
    "lead": "lead",
}


def normalize_mobile(mobile: str) -> str:
    digits = "".join(char for char in str(mobile) if char.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def generate_store_delivery_label_code(
    customer_id: str,
    customer_name: str,
    customer_segment: str,
    mobile: str,
) -> dict:
    safe_mobile = normalize_mobile(mobile)
    safe_name = (customer_name or "customer").strip().upper()
    safe_segment = (customer_segment or "client").strip().lower()
    created_at = datetime.now(timezone.utc)

    seed = f"{customer_id}|{safe_name}|{safe_segment}|{safe_mobile}|{created_at.strftime('%Y%m%d%H%M')}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6].upper()
    segment_prefix = (safe_segment[:1] or "C").upper()
    mobile_suffix = (safe_mobile[-4:] if safe_mobile else "0000")
    label_code = f"TSL-{segment_prefix}{mobile_suffix}-{digest}"

    return {
        "label_code": label_code,
        "created_at_utc": created_at.isoformat(),
        "customer_name": safe_name,
        "mobile": safe_mobile,
        "customer_segment": safe_segment,
    }


def _crm_get(path: str, params: dict | None = None) -> dict:
    response = httpx.get(f"{CRM_BASE_URL}/{path}", params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    return response.json()


def classify_customer_profile(mobile: str) -> dict:
    """Single-call customer classification using the /clienttype API endpoint."""
    mobile = normalize_mobile(mobile)
    result = {
        "mobile": mobile,
        "segment": None,
        "customer_name": None,
        "reason": None,
        "orders_count": 0,
        "active_orders": 0,
    }

    try:
        data = _crm_get("clienttype.php", {"mobile": mobile})
    except Exception as exc:
        result["segment"] = "error"
        result["reason"] = f"clienttype api failed: {exc}"
        return result

    if data.get("status") != "success":
        result["segment"] = "new_user"
        result["reason"] = "mobile number not found in CRM"
        return result

    client = data.get("client") or {}
    result["customer_name"] = client.get("cname")
    result["orders_count"] = data.get("total_orders", 0)

    crm_type = data.get("type", "")
    result["segment"] = _TYPE_TO_SEGMENT.get(crm_type, "new_user")
    result["reason"] = data.get("description", "")

    orders = data.get("orders", [])
    result["active_orders"] = len(orders)

    return result


def fetch_client_addresses(mobile: str) -> list[dict]:
    """Fetch all saved addresses for a customer from CRM. Returns a list of address dicts."""
    mobile = normalize_mobile(mobile)
    india_mobile = f"91{mobile}" if not mobile.startswith("91") else mobile
    # Try both normalized and india-prefixed mobile
    for candidate in [mobile, india_mobile]:
        try:
            data = _crm_get("clientaddress.php", {"mobile": candidate})
            if not data:
                continue
            # Response may be a list directly or wrapped under a key
            if isinstance(data, list):
                return data
            for key in ("addresses", "data", "address"):
                if isinstance(data.get(key), list):
                    return data[key]
            # Single address object
            if data.get("status") == "success" and data.get("address_id"):
                return [data]
        except Exception:
            continue
    return []


def format_address_label(addr: dict, index: int | None = None) -> str:
    """Format a single address dict into a readable one-line label."""
    parts = [
        addr.get("address1"),
        addr.get("address2"),
        addr.get("locality"),
        addr.get("city"),
        addr.get("pincode"),
    ]
    label = ", ".join(str(p).strip() for p in parts if str(p or "").strip())
    if not label:
        label = "Address not available"
    prefix = f"{index}. " if index is not None else ""
    return f"{prefix}{label}"


def fetch_customer_info_data(mobile: str) -> dict:
    mobile = normalize_mobile(mobile)
    try:
        data = _crm_get("clientinfo.php", {"mobile": mobile})
        if data.get("status") == "success" and data.get("client"):
            return data["client"]
    except Exception:
        return {}
    return {}


def format_client_address(client: dict) -> str:
    address_fields = [
        client.get("address1"),
        client.get("address2"),
        client.get("locality"),
        client.get("city"),
        client.get("pincode"),
    ]
    address_parts = [str(value).strip() for value in address_fields if str(value or "").strip()]
    if address_parts:
        return ", ".join(address_parts)

    fallback = str(client.get("address") or "").strip()
    return fallback or "Address not available"


def fetch_customer_info(mobile: str) -> str:
    mobile = normalize_mobile(mobile)
    try:
        client = fetch_customer_info_data(mobile)
        if client:
            return (
                f"Customer: {client.get('cname', 'N/A')} ({mobile})\n"
                f"Email: {client.get('email', 'N/A')}\n"
                f"Phone: {client.get('tel', 'N/A')}\n"
                f"Address: {format_client_address(client)}"
            )
        return "Customer not found in our system."
    except Exception as exc:
        return f"Error fetching customer info: {exc}"


def fetch_customer_whatsapp_history(mobile: str) -> str:
    mobile = normalize_mobile(mobile)
    try:
        data = _crm_get("clientwhatsapp.php", {"mobile": mobile})
        messages = data.get("data") if isinstance(data, dict) else data
        if isinstance(messages, list) and messages:
            history = "\n".join(
                f"{msg.get('date', '')}: {msg.get('message', '')}"
                for msg in messages[-5:]
            )
            return f"Recent WhatsApp history:\n{history}"
        return "No WhatsApp history available."
    except Exception as exc:
        return f"Error fetching WhatsApp history: {exc}"


def fetch_customer_orders(mobile: str) -> str:
    mobile = normalize_mobile(mobile)
    try:
        data = _crm_get("clientorders.php", {"mobile": mobile})
        if data.get("status") == "success" and data.get("orders"):
            orders = []
            for order in data["orders"][:5]:
                order_id = order.get("id", "N/A")
                stage_label = order.get("stage_label") or order.get("stage", "N/A")
                order_date = order.get("rdate", "N/A")
                orders.append(f"Order #{order_id}: {stage_label} (Ordered: {order_date})")
            return f"Your orders ({data.get('total_orders', len(orders))} total):\n" + "\n".join(orders)
        return "No orders found for this customer."
    except Exception as exc:
        return f"Error fetching orders: {exc}"


def fetch_items_info() -> str:
    try:
        data = _crm_get("items.php")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list) and items:
            lines = [f"{item.get('iname', item.get('name', 'N/A'))} (ID: {item.get('id', 'N/A')})" for item in items[:10]]
            return "Available items:\n" + "\n".join(lines)
        return "No items available."
    except Exception as exc:
        return f"Error fetching items: {exc}"


def fetch_items_by_category(category_id: str) -> str:
    try:
        data = _crm_get("items.php", {"catid": category_id})
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list) and items:
            lines = [f"{item.get('iname', item.get('name', 'N/A'))} - ₹{item.get('price', 'N/A')}" for item in items]
            return f"Items in category {category_id}:\n" + "\n".join(lines)
        return "No items found in this category."
    except Exception as exc:
        return f"Error fetching category items: {exc}"


def fetch_items_with_price() -> str:
    try:
        data = _crm_get("itemswithprice.php")
        if data.get("status") == "success" and data.get("data"):
            items = []
            for item in data["data"][:10]:
                name = item.get("iname", "N/A")
                subitems = item.get("subitems") or []
                if subitems:
                    subitem_prices = [
                        f"{sub.get('subitem_name', '')}: ₹{sub.get('price', '0')}"
                        for sub in subitems[:2]
                    ]
                    items.append(f"{name}\n  " + "\n  ".join(subitem_prices))
                else:
                    items.append(f"{name}: ₹{item.get('price', 'N/A')}")
            return "Available services & prices:\n\n" + "\n\n".join(items)
        return "No items with pricing available."
    except Exception as exc:
        return f"Error fetching item prices: {exc}"


def fetch_customer_measurements(mobile: str) -> str:
    mobile = normalize_mobile(mobile)
    try:
        data = _crm_get("clientmeasurements.php", {"mobile": mobile})
        if data.get("status") != "success":
            return "Unable to fetch measurements."

        measurements = []
        if data.get("women_forms", {}).get("data"):
            for form in data["women_forms"]["data"][:3]:
                measurements.append(
                    f"Women's Form #{form.get('id', 'N/A')}: Bust {form.get('bust', 'N/A')}, Waist {form.get('blouse_waist', 'N/A')} (Created: {form.get('created_at', 'N/A')})"
                )
        if data.get("men_forms", {}).get("data"):
            for form in data["men_forms"]["data"][:3]:
                measurements.append(
                    f"Men's Form #{form.get('id', 'N/A')}: Chest {form.get('chest', 'N/A')}, Waist {form.get('waist', 'N/A')} (Created: {form.get('created_at', 'N/A')})"
                )

        if measurements:
            return "Your measurement forms:\n" + "\n".join(measurements)
        return "No measurement forms found. Would you like to submit measurements for a new order?"
    except Exception as exc:
        return f"Error fetching measurements: {exc}"


def fetch_customer_measurement_forms(mobile: str) -> list[dict]:
    mobile = normalize_mobile(mobile)
    try:
        data = _crm_get("clientmeasurements.php", {"mobile": mobile})
        if data.get("status") != "success":
            return []

        forms: list[dict] = []
        if data.get("women_forms", {}).get("data"):
            for form in data["women_forms"]["data"][:5]:
                form_id = form.get("id", "N/A")
                forms.append({
                    "label": f"Women's Form #{form_id}",
                    "details": (
                        f"Women's Form #{form_id}\n"
                        f"Bust: {form.get('bust', 'N/A')}\n"
                        f"Waist: {form.get('blouse_waist', 'N/A')}\n"
                        f"Created: {form.get('created_at', 'N/A')}"
                    ),
                })

        if data.get("men_forms", {}).get("data"):
            for form in data["men_forms"]["data"][:5]:
                form_id = form.get("id", "N/A")
                forms.append({
                    "label": f"Men's Form #{form_id}",
                    "details": (
                        f"Men's Form #{form_id}\n"
                        f"Chest: {form.get('chest', 'N/A')}\n"
                        f"Waist: {form.get('waist', 'N/A')}\n"
                        f"Created: {form.get('created_at', 'N/A')}"
                    ),
                })

        return forms
    except Exception:
        return []



PICKUP_TIME_SLOTS = {
    "1": {"label": "Morning (9AM – 2PM)", "value": 1},
    "2": {"label": "Afternoon (2PM – 9PM)", "value": 2},
}

STORE_VISIT_TIME_SLOTS = {
    "1": "11 AM - 2PM",
    "2": "3 PM - 7PM",
}


def schedule_pickup_crm(mobile: str, pickup_date: str, time_slot: int) -> dict:
    """Call the CRM schedule pickup API.
    pickup_date: YYYY-MM-DD format
    time_slot: 1 = Morning 9AM-2PM, 2 = Afternoon 2PM-9PM
    """
    raw_mobile_digits = "".join(char for char in str(mobile) if char.isdigit())
    normalized_mobile = normalize_mobile(raw_mobile_digits)
    india_mobile = f"91{normalized_mobile}" if not normalized_mobile.startswith("91") else normalized_mobile

    mobile_candidates: list[str] = []
    for candidate in [raw_mobile_digits, india_mobile, normalized_mobile]:
        if candidate and candidate not in mobile_candidates:
            mobile_candidates.append(candidate)

    def _is_success(data: dict) -> bool:
        return str(data.get("status", "")).strip().lower() == "success"

    try:
        endpoint = f"{CRM_BASE_URL}/schedulepickup.php"
        final_data = None
        final_payload = None
        final_http_status = None
        for candidate_mobile in mobile_candidates:
            payload = {
                "mobile": candidate_mobile,
                "pickup_date": pickup_date,
                "pickup_time": time_slot,
            }
            response = httpx.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            final_http_status = response.status_code
            try:
                data = response.json() if response.text else {}
            except ValueError:
                data = {"raw": response.text}
            final_data = data
            final_payload = payload
            if _is_success(data):
                break

        success = _is_success(final_data) if final_data else False
        pickup_data = (final_data or {}).get("data") if isinstance(final_data, dict) else None
        return {
            "success": success,
            "message": str((final_data or {}).get("message") or ("Pickup scheduled successfully." if success else "Could not confirm pickup scheduling.")).strip(),
            "payload": final_payload,
            "response": final_data,
            "pickup": pickup_data if isinstance(pickup_data, dict) else None,
            "http_status": final_http_status,
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to schedule pickup: {exc}",
            "payload": None,
        }


def book_store_appointment(mobile: str, bookdate: str, booktime: str, store_id: int = 1) -> dict:
    """Book a store appointment using CRM API."""
    raw_mobile_digits = "".join(char for char in str(mobile) if char.isdigit())
    normalized_mobile = normalize_mobile(raw_mobile_digits)
    india_mobile = f"91{normalized_mobile}" if not normalized_mobile.startswith("91") else normalized_mobile

    mobile_candidates: list[str] = []
    for candidate in [raw_mobile_digits, india_mobile, normalized_mobile]:
        if candidate and candidate not in mobile_candidates:
            mobile_candidates.append(candidate)

    try:
        endpoint = f"{CRM_BASE_URL}/bookappointment.php"
        final_data = None
        final_payload = None
        final_http_status = None

        for candidate_mobile in mobile_candidates:
            payload = {
                "mobile": candidate_mobile,
                "store_id": int(store_id),
                "bookdate": str(bookdate).strip(),
                "booktime": str(booktime).strip(),
            }
            response = httpx.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            final_http_status = response.status_code
            try:
                data = response.json() if response.text else {}
            except ValueError:
                data = {"raw": response.text}
            final_data = data
            final_payload = payload
            if str(data.get("status", "")).strip().lower() == "success":
                break

        success = str((final_data or {}).get("status", "")).strip().lower() == "success"
        appointment_data = (final_data or {}).get("data") if isinstance(final_data, dict) else None
        return {
            "success": success,
            "message": str((final_data or {}).get("message") or ("Appointment booked successfully." if success else "Could not confirm appointment booking.")).strip(),
            "payload": final_payload,
            "response": final_data,
            "appointment": appointment_data if isinstance(appointment_data, dict) else None,
            "http_status": final_http_status,
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to book appointment: {exc}",
            "payload": None,
        }


def fetch_client_appointment_history(mobile: str) -> list[dict]:
    """Fetch appointment history for a mobile from CRM."""
    mobile = normalize_mobile(mobile)
    india_mobile = f"91{mobile}" if not mobile.startswith("91") else mobile

    for candidate in [india_mobile, mobile]:
        try:
            data = _crm_get("bookappointment.php", {"mobile": candidate})
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                history = data.get("data") or data.get("appointments") or data.get("history")
                if isinstance(history, list):
                    return history
                if str(data.get("status", "")).strip().lower() == "success" and isinstance(data.get("data"), dict):
                    return [data.get("data")]
        except Exception:
            continue
    return []


def add_client_address(mobile: str, address_payload: dict) -> dict:
    raw_mobile_digits = "".join(char for char in str(mobile) if char.isdigit())
    normalized_mobile = normalize_mobile(raw_mobile_digits)

    mobile_candidates: list[str] = []
    if raw_mobile_digits:
        mobile_candidates.append(raw_mobile_digits)
    if normalized_mobile and normalized_mobile not in mobile_candidates:
        mobile_candidates.append(normalized_mobile)
    india_mobile = f"91{normalized_mobile}" if normalized_mobile else ""
    if india_mobile and india_mobile not in mobile_candidates:
        mobile_candidates.append(india_mobile)

    base_payload = {
        "action": "add",
        "address1": str(address_payload.get("address1", "")).strip(),
        "address2": str(address_payload.get("address2", "")).strip(),
        "locality": str(address_payload.get("locality", "")).strip(),
        "city": str(address_payload.get("city", "")).strip(),
        "pincode": str(address_payload.get("pincode", "")).strip(),
        "lat": str(address_payload.get("lat", "")).strip(),
        "lng": str(address_payload.get("lng", "")).strip(),
    }

    def _extract_response_data(resp: httpx.Response) -> dict:
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                return parsed
            return {"raw": parsed}
        except ValueError:
            return {"raw": resp.text}

    def _is_success(data: dict) -> bool:
        status = str(data.get("status", "")).strip().lower()
        if status == "success":
            return True
        if status in {"error", "failed", "failure"}:
            return False

        message = str(data.get("message", "")).strip().lower()
        if any(token in message for token in ["saved", "updated", "success"]):
            return True
        if any(token in message for token in ["error", "failed", "invalid", "required"]):
            return False
        return False

    try:
        endpoint = f"{CRM_BASE_URL}/clientaddress.php"
        attempts: list[tuple[str, dict]] = []
        final_payload = None
        final_data = None

        for candidate_mobile in mobile_candidates:
            payload = {
                "mobile": candidate_mobile,
                **base_payload,
            }
            response = httpx.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            data = _extract_response_data(response)
            attempts.append((candidate_mobile, data))
            final_payload = payload
            final_data = data
            if _is_success(data):
                break

        if final_payload is None or final_data is None:
            return {
                "success": False,
                "message": "Could not submit address payload to CRM.",
                "payload": {"mobile": raw_mobile_digits, **base_payload},
            }

        success = _is_success(final_data)

        return {
            "success": success,
            "message": str(final_data.get("message") or ("Address saved successfully." if success else "Could not confirm address save in CRM.")).strip(),
            "payload": final_payload,
            "response": final_data,
            "attempted_mobiles": [mobile for mobile, _ in attempts],
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to save address: {exc}",
            "payload": {"mobile": raw_mobile_digits, **base_payload},
        }


def update_client_address(mobile: str, address_id: int | str, address_payload: dict) -> dict:
    raw_mobile_digits = "".join(char for char in str(mobile) if char.isdigit())
    normalized_mobile = normalize_mobile(raw_mobile_digits)

    mobile_candidates: list[str] = []
    if raw_mobile_digits:
        mobile_candidates.append(raw_mobile_digits)
    if normalized_mobile and normalized_mobile not in mobile_candidates:
        mobile_candidates.append(normalized_mobile)
    india_mobile = f"91{normalized_mobile}" if normalized_mobile else ""
    if india_mobile and india_mobile not in mobile_candidates:
        mobile_candidates.append(india_mobile)

    base_payload = {
        "action": "update",
        "address_id": str(address_id).strip(),
        "address1": str(address_payload.get("address1", "")).strip(),
        "address2": str(address_payload.get("address2", "")).strip(),
        "locality": str(address_payload.get("locality", "")).strip(),
        "city": str(address_payload.get("city", "")).strip(),
        "pincode": str(address_payload.get("pincode", "")).strip(),
        "lat": str(address_payload.get("lat", "")).strip(),
        "lng": str(address_payload.get("lng", "")).strip(),
    }

    def _extract_response_data(resp: httpx.Response) -> dict:
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                return parsed
            return {"raw": parsed}
        except ValueError:
            return {"raw": resp.text}

    def _is_success(data: dict) -> bool:
        status = str(data.get("status", "")).strip().lower()
        if status == "success":
            return True
        if status in {"error", "failed", "failure"}:
            return False

        message = str(data.get("message", "")).strip().lower()
        if any(token in message for token in ["saved", "updated", "success"]):
            return True
        if any(token in message for token in ["error", "failed", "invalid", "required"]):
            return False
        return False

    try:
        endpoint = f"{CRM_BASE_URL}/clientaddress.php"
        attempts: list[tuple[str, dict]] = []
        final_payload = None
        final_data = None

        for candidate_mobile in mobile_candidates:
            payload = {
                "mobile": candidate_mobile,
                **base_payload,
            }
            response = httpx.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = _extract_response_data(response)
            attempts.append((candidate_mobile, data))
            final_payload = payload
            final_data = data
            if _is_success(data):
                break

        if final_payload is None or final_data is None:
            return {
                "success": False,
                "message": "Could not submit address update payload to CRM.",
                "payload": {"mobile": raw_mobile_digits, **base_payload},
            }

        success = _is_success(final_data)

        return {
            "success": success,
            "message": str(final_data.get("message") or ("Address updated successfully." if success else "Could not confirm address update in CRM.")).strip(),
            "payload": final_payload,
            "response": final_data,
            "attempted_mobiles": [mobile for mobile, _ in attempts],
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to update address: {exc}",
            "payload": {"mobile": raw_mobile_digits, **base_payload},
        }


@tool
def classify_customer(mobile: str) -> dict:
    """Identify whether a WhatsApp number belongs to an active, existing, or new customer."""
    return classify_customer_profile(mobile)


@tool
def get_customer_info(mobile: str) -> str:
    """Get customer information from CRM."""
    return fetch_customer_info(mobile)


@tool
def get_customer_whatsapp_history(mobile: str) -> str:
    """Get customer's WhatsApp conversation history."""
    return fetch_customer_whatsapp_history(mobile)


@tool
def get_customer_orders(mobile: str) -> str:
    """Get customer's order information."""
    return fetch_customer_orders(mobile)


@tool
def get_items_info() -> str:
    """Get all available items."""
    return fetch_items_info()


@tool
def get_items_by_category(category_id: str) -> str:
    """Get items by category."""
    return fetch_items_by_category(category_id)


@tool
def get_items_with_price() -> str:
    """Get all items with pricing information."""
    return fetch_items_with_price()


@tool
def get_customer_measurements(mobile: str) -> str:
    """Get customer's measurement forms."""
    return fetch_customer_measurements(mobile)