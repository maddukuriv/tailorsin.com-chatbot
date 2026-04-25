import json
import os
import re
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from app.tools import (
    add_client_address,
    book_store_appointment,
    fetch_client_appointment_history,
    update_client_address,
    schedule_pickup_crm,
    PICKUP_TIME_SLOTS,
    fetch_client_addresses,
    format_address_label,
    fetch_customer_info,
    fetch_customer_info_data,
    fetch_customer_measurements,
    fetch_customer_measurement_forms,
    fetch_customer_orders,
    fetch_customer_whatsapp_history,
    fetch_items_info,
    fetch_items_with_price,
    format_client_address,
    generate_store_delivery_label_code,
    normalize_mobile,
)

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    sentiment_score: int
    needs_human: bool
    context: Dict[str, Any]
    conversation_id: str


TAILORSIN_OVERVIEW = (
    "tailorsin.com is a premium bespoke tailoring service offering complimentary pickup and delivery across Hyderabad for orders above ₹500.\n\n"
    
    "We provide custom stitching and alterations for women's, men's, and kids' wear, ensuring a smooth and hassle-free experience from pickup to delivery.\n\n"
    
    "How the process works:\n\n"
    
    "1. Schedule a pickup at your convenience.\n"
    "2. Hand over your fabric along with a sample or reference design.\n"
    "3. Within 6 business hours, our team will contact you to confirm design details and share a detailed estimate.\n"
    "4. Once you approve the estimate and complete the payment, we begin production and confirm the delivery timeline.\n"
    "5. If you choose not to proceed, your unstitched fabric will be safely returned.\n"
    "6. After production, an e-invoice will be generated.\n"
    "7. Your stitched outfits will be delivered to your doorstep.\n"
    "8. Free alterations are available within 7 days of delivery.\n\n"
    
    "Our goal is to make tailoring simple, convenient, and perfectly tailored to your needs."
)

ITEMS_BROWSE_INTRO = (
    "We stitch Women's, Men's, and Kids wear with custom fitting, design preferences, and finishing support. Here are the categories currently available:\n\n"
)

MEASUREMENT_POLICY = (
    "Currently, we do not offer home measurement services due to quality control and customer privacy reasons. "
    "To ensure a reliable fit, please share a sample or reference outfit during pickup, or book a store visit by appointment."
)

DELIVERY_TIMELINE = (
    "Most orders are completed within 24 hours of cloth pickup once the design and estimate are approved. "
    "For complex garments, detailed embroidery, or special finishing, our team confirms the committed delivery date during order approval."
)

SERVICE_AREAS = (
    "We currently serve Hyderabad. Share your area or pincode and we will confirm pickup and delivery availability for your location before scheduling."
)

OWN_FABRIC_POLICY = (
    "Yes, absolutely. You can use your own fabric, and our team can pick it up from your location and return the finished garment once stitching is complete."
)

STORE_VISIT_INFO = (
    "Certainly 😊 Please let us know in advance so we can schedule an appointment for your visit.\n\n"
    "2nd Floor, Door No 8-2-293/82/A/16, Road No. 5,\n"
    "Next to Jubilee Hills Road No. 5 Metro Station,\n"
    "Jubilee Hills, Hyderabad, Telangana – 500033\n\n"
    "🕙 Timings:\n"
    "• Monday to Saturday: 10 AM – 8 PM\n"
    "• Sunday: 11 AM – 6 PM\n\n"
    "Visits are by appointment basis.\n"
    "📞 For address-related queries, call: 9966891000"
)

PLACE_ORDER_DETAILS = (
    "Great 😊 For scheduling pickup, please share the following details:\n\n"
    "1️⃣ Full Name\n"
    "2️⃣ Email Address\n"
    "3️⃣ First and Second Preferred Language of Communication\n"
    "4️⃣ Pickup or Delivery Address with Pincode and Geo Location\n"
    "5️⃣ Calling Number (if different from WhatsApp number)\n\n"
    "Please note:\n"
    "Due to quality control and customer privacy reasons, we currently do not offer home measurement services.\n\n"
    "Can you confirm if you can provide a sample or reference outfit at the time of pickup?"
)

REGISTER_RESPONSE = (
    "To get started, please share your name, area, and preferred pickup time. "
    "If you're ready to place an order immediately, you can also choose option 10 and share the full pickup details so our team can coordinate faster."
)

ORDER_CHANGE_GUIDANCE = (
    "You can request design, size, delivery, or item changes while your order is in progress. "
    "Please reply with the exact change details, and our team will review feasibility and confirm the next step with you."
)

ALTERATIONS_GUIDANCE = (
    "We offer alteration support for completed garments. Please share what needs to be altered, and our team will guide you on pickup, timelines, and any applicable charges if outside the free alteration window."
)

ORDER_SUPPORT_TRANSFER = (
    "I’m connecting you to our order support team now so they can assist you directly with this request."
)

TAILORING_SPECIALIST_TRANSFER = (
    "I’m connecting you to a tailorsin.com specialist now so you can continue with a team member directly."
)

CLIENT_PICKUP_CHOICE_PROMPT = (
    "1. Would you like us to schedule pickup from your saved address\n"
    "2. Add a new address and schedule pickup"
)

CLIENT_PICKUP_TIME_PROMPT = (
    "Please select your preferred pickup time slot:\n"
    "1. Morning (9AM – 2PM)\n"
    "2. Afternoon (2PM – 9PM)"
)

CLIENT_PICKUP_DATE_PROMPT = (
    "Please select your pickup date:\n"
    "1. Today\n"
    "2. Tomorrow\n"
    "3. Enter another date (YYYY-MM-DD)"
)

CLIENT_VISIT_DATE_PROMPT = (
    "Please select your store visit date:\n"
    "1. Today\n"
    "2. Tomorrow\n"
    "3. Enter another date (YYYY-MM-DD)"
)

CLIENT_NEW_ADDRESS_PROMPT = (
    "Please provide the pickup postal address with pincode, nearest landmark, city, and geo coordinates in this format:\n"
    "Address line 1,\n"
    "Address line 2,\n"
    "Locality,\n"
    "City,\n"
    "Pincode,\n"
    "Latitude, Longitude"
)

CLIENT_VISIT_PROMPT = (
    "Please share your preferred visit date, time, and city. Our team will confirm the appointment shortly.\n\n"
    + STORE_VISIT_INFO
)

CLIENT_ESTIMATE_PROMPT = (
    "Share your reference picture along with your standard size, height, and desired dress length in inches."
)

CLIENT_ESTIMATE_FABRIC_PROMPT = "Do you have any fabric preference?"

CLIENT_ESTIMATE_HUMAN_TRANSFER = "Our team will reach out to you shortly."

CLIENT_LATER_RESPONSE = "Thanks for enquiring with us. Have a great day."

CLIENT_GENERAL_HELP_PROMPT = "Please tell us what you need help with and our team will guide you further."
CLIENT_SMART_HELP_PROMPT = (
    "Please share your issue in one message, for example: order delay, payment issue, alteration request, pickup problem, or anything else."
)
HELP_CATEGORY_TEMPLATES = {
    "order_delay": "Thanks for reporting the delay. I have logged this with priority and our team will update you on the current order status shortly.",
    "payment_issue": "I understand this payment concern. Please share the payment date, amount, and transaction reference so we can verify and resolve it quickly.",
    "alteration_issue": "Thanks for sharing the alteration issue. Please describe the exact fit/change needed and we will guide you with the fastest correction path.",
    "pickup_issue": "I have captured the pickup issue. Please confirm your preferred pickup window and full address so we can re-schedule promptly.",
    "delivery_issue": "I understand the delivery issue. I have flagged this for immediate review and our support team will update you soon.",
    "quality_issue": "Thank you for raising this quality concern. Please share clear photos and your order details so we can resolve this on priority.",
    "billing_issue": "I have noted your billing issue. Please share the invoice number and billing concern so we can validate and correct it quickly.",
    "other": "Thanks for sharing the details. I have captured your request and our team will guide you with the right next step.",
}
HELP_CATEGORY_AUTO_ESCALATE = {"payment_issue", "delivery_issue", "quality_issue", "billing_issue"}
STORE_DELIVERY_ADDRESS = (
    "2nd Floor, Door No 8-2-293/82/A/16, Road No. 5, "
    "Next to Jubilee Hills Road No. 5 Metro Station, Jubilee Hills, Hyderabad, Telangana - 500033"
)
STORE_DELIVERY_PHONE = "9966891000"

class E_TailoringAgent:
    def __init__(self, provider: str, api_key: str, model: str | None = None):
        normalized_provider = provider.lower().strip()

        if normalized_provider == "groq":
            self.llm = ChatGroq(
                model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.3,
                api_key=api_key,
            )
        elif normalized_provider == "openai":
            self.llm = ChatOpenAI(
                model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                api_key=api_key,
            )
        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(state_schema=AgentState)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("process_intent", self.process_intent)
        workflow.add_node("generate_response", self.generate_response)

        workflow.set_entry_point("analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "process_intent")
        workflow.add_edge("process_intent", "generate_response")
        workflow.set_finish_point("generate_response")

        return workflow.compile()

    async def summarize_handoff(
        self,
        messages: List[Dict[str, str]],
        customer_name: str | None = None,
        customer_segment: str | None = None,
    ) -> str:
        transcript_lines = []
        for message in messages[-12:]:
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            transcript_lines.append(f"{role}: {content}")

        transcript = "\n".join(transcript_lines)
        prompt = (
            "Summarize this WhatsApp support conversation for a human tailoring support agent. "
            "Keep it short and operational.\n\n"
            f"Customer name: {customer_name or 'Unknown'}\n"
            f"Customer segment: {customer_segment or 'Unknown'}\n\n"
            "Return plain text with exactly these headings:\n"
            "Customer\n"
            "Reason for handoff\n"
            "Key details\n"
            "Recommended next step\n\n"
            "Conversation:\n"
            f"{transcript}"
        )

        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]["content"]
        prompt = (
            f"Analyze the sentiment of this customer message for a tailoring company. "
            f"Only set wants_human to true if the customer explicitly asks for a human representative, "
            f"is strongly upset, or is clearly complaining. "
            f"Return valid JSON with keys: score (1-10), wants_human (true/false), reason.\n"
            f"Message: {json.dumps(last_message)}\n"
            f"Output example: {{\"score\": 8, \"wants_human\": false, \"reason\": \"Friendly and conversational\"}}"
        )

        response = await self.llm.ainvoke(prompt)
        sentiment_data = self._safe_parse_json(response.content)

        state["sentiment_score"] = sentiment_data.get("score", 5)
        wants_human = sentiment_data.get("wants_human") if isinstance(sentiment_data.get("wants_human"), bool) else False
        explicit_human_request = any(
            token in last_message.lower()
            for token in ["human", "real person", "agent", "customer care", "representative", "someone"]
        )

        state["needs_human"] = wants_human or explicit_human_request
        state["context"]["sentiment_reason"] = sentiment_data.get("reason", "")
        return state    

    async def process_intent(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]["content"].strip()
        mobile = normalize_mobile(state["conversation_id"])
        customer_segment = state["context"].get("customer_segment", "new_user")

        if customer_segment in {"client", "lead"}:
            client_flow_state = await self._handle_client_segment_flow(state, last_message, mobile)
            if client_flow_state is not None:
                return client_flow_state

        if last_message.isdigit() and last_message not in ["0", "00"] and last_message not in [str(i) for i in range(1, 11)]:
            state["context"]["intent"] = "invalid_menu_option"
            state["context"]["tool_response"] = (
                "I didn't catch that option. Please reply with a number from 1 to 10, "
                "reply 0 for the main menu, or 00 for a human agent."
            )
            return state

        if last_message in [str(i) for i in range(1, 11)]:
            segment_menu_handlers = {
                "active_client": {
                    "1": ("order_status", lambda: fetch_customer_orders(mobile)),
                    "2": ("order_ready_time", lambda: "Here are the latest details for your active order:\n\n" + fetch_customer_orders(mobile)),
                    "3": ("order_changes", lambda: ORDER_CHANGE_GUIDANCE),
                    "4": ("tracking", lambda: "Here is your latest order and delivery update:\n\n" + fetch_customer_orders(mobile)),
                    "5": ("measurements", lambda: fetch_customer_measurements(mobile)),
                    "6": ("order_history", lambda: fetch_customer_orders(mobile)),
                    "7": ("add_items", lambda: "You can add another item to your ongoing order or create a parallel order. Here is our current price list:\n\n" + fetch_items_with_price()),
                    "8": ("pricing", lambda: fetch_items_with_price()),
                },
                "client": {
                    "1": ("new_order", lambda: PLACE_ORDER_DETAILS),
                    "2": ("browse_items", lambda: ITEMS_BROWSE_INTRO + fetch_items_info()),
                    "3": ("pricing", lambda: fetch_items_with_price()),
                    "4": ("order_history", lambda: fetch_customer_orders(mobile)),
                    "5": ("measurements", lambda: fetch_customer_measurements(mobile)),
                    "6": ("alterations", lambda: ALTERATIONS_GUIDANCE),
                    "7": ("turnaround_time", lambda: DELIVERY_TIMELINE),
                    "8": ("book_visit", lambda: STORE_VISIT_INFO),
                    "9": ("own_fabric", lambda: OWN_FABRIC_POLICY),
                    "10": ("human_support", lambda: TAILORING_SPECIALIST_TRANSFER),
                },
                "lead": {
                    "1": ("schedule_pickup", lambda: CLIENT_PICKUP_CHOICE_PROMPT),
                    "2": ("price_catalogue", lambda: fetch_items_with_price()),
                    "3": ("book_visit", lambda: CLIENT_VISIT_PROMPT),
                    "4": ("estimate", lambda: CLIENT_ESTIMATE_PROMPT),
                    "5": ("measurements", lambda: fetch_customer_measurements(mobile)),
                    "6": ("store_address", lambda: STORE_VISIT_INFO),
                    "7": ("fabric_delivery", lambda: OWN_FABRIC_POLICY),
                    "8": ("general_help", lambda: TAILORSIN_OVERVIEW),
                },
                "new_user": {
                    "1": ("about_tailorsin", lambda: TAILORSIN_OVERVIEW),
                    "2": ("browse_items", lambda: ITEMS_BROWSE_INTRO + fetch_items_info()),
                    "3": ("pricing", lambda: fetch_items_with_price()),
                    "4": ("measurement_process", lambda: MEASUREMENT_POLICY),
                    "5": ("delivery_speed", lambda: DELIVERY_TIMELINE),
                    "6": ("service_areas", lambda: SERVICE_AREAS),
                    "7": ("book_visit", lambda: STORE_VISIT_INFO),
                    "8": (
                        "register_first_order",
                        lambda: "Registration and first-order placement will be enabled soon. Please use option 9 if you would like our team to assist you right away.",
                    ),
                },
            }
            menu_handlers = segment_menu_handlers.get(customer_segment, segment_menu_handlers["new_user"])
            if last_message not in menu_handlers:
                state["context"]["intent"] = "reserved_navigation_option"
                state["context"]["tool_response"] = "Please use option 9 to chat with a human agent or option 10 to go back to the main menu."
                return state
            intent, tool_func = menu_handlers[last_message]
            state["context"]["intent"] = intent
            state["context"]["tool_response"] = tool_func()
            if intent == "human_support":
                state["needs_human"] = True
            return state

        # Continue with existing intent classification for non-numbered messages
        prompt = (
            f"Classify this WhatsApp customer request for a tailoring company. "
            f"Return valid JSON with keys: intent, order_id, garment, fabric, customizations, wants_human.\n"
            f"If the customer is simply asking a normal question, set wants_human to false.\n"
            f"Possible intents: order_status, fabric_availability, pricing, measurements, greeting, complaint, general.\n"
            f"Message: {json.dumps(last_message)}\n"
            f"Output example: {{\"intent\": \"order_status\", \"order_id\": \"12345\", \"garment\": \"shirt\", \"fabric\": null, \"customizations\": [], \"wants_human\": false}}"
        )

        response = await self.llm.ainvoke(prompt)
        intent_data = self._safe_parse_json(response.content)
        if isinstance(intent_data, dict):
            state["context"].update(intent_data)
        else:
            intent_data = {}

        intent = intent_data.get("intent", "general")
        if intent == "order_status":
            state["context"]["tool_response"] = fetch_customer_orders(mobile)
        elif intent == "measurements":
            state["context"]["tool_response"] = fetch_customer_measurements(mobile)
        elif intent == "pricing":
            state["context"]["tool_response"] = fetch_items_with_price()
        elif intent == "general" and customer_segment in ("client", "active_client"):
            state["context"]["tool_response"] = fetch_customer_info(mobile)

        if intent_data.get("wants_human") is True:
            state["needs_human"] = True

        return state

    async def _handle_client_segment_flow(self, state: AgentState, last_message: str, mobile: str) -> AgentState | None:
        pending_flow = state["context"].get("client_flow")
        normalized = "".join(last_message.lower().split())

        if pending_flow == "pickup_choice":
            return await self._handle_client_pickup_choice(state, last_message, normalized, mobile)
        if pending_flow == "select_saved_address":
            return await self._handle_client_select_saved_address(state, last_message, normalized, mobile)
        if pending_flow == "awaiting_new_address":
            return await self._handle_client_new_address(state, last_message, mobile)
        if pending_flow == "pickup_date_selection":
            return await self._handle_client_pickup_date_choice(state, last_message, normalized)
        if pending_flow == "pickup_time_slot":
            return await self._handle_client_pickup_time_slot(state, last_message, normalized, mobile)
        if pending_flow == "store_visit_date_selection":
            return await self._handle_client_store_visit_date_choice(state, last_message, normalized, mobile)
        if pending_flow == "store_visit_exact_slot_selection":
            return await self._handle_client_store_visit_exact_slot_choice(state, normalized, mobile)
        if pending_flow == "pricing_followup":
            return self._handle_client_pricing_followup(state, normalized)
        if pending_flow == "estimate_details":
            state["context"]["estimate_request_details"] = last_message
            state["context"]["client_flow"] = "estimate_fabric_preference"
            state["context"]["intent"] = "estimate_fabric_requirement"
            state["context"]["tool_response"] = CLIENT_ESTIMATE_FABRIC_PROMPT
            return state
        if pending_flow == "estimate_fabric_preference":
            state["context"]["estimate_fabric_preference"] = last_message
            state["context"]["client_flow"] = None
            state["context"]["intent"] = "estimate_submitted"
            state["context"]["tool_response"] = CLIENT_ESTIMATE_HUMAN_TRANSFER
            state["context"]["handoff_after_response"] = True
            return state
        if pending_flow == "measurement_choice":
            return self._handle_client_measurement_choice(state, normalized)
        if pending_flow == "shipping_label_confirmation":
            return self._handle_client_shipping_label_confirmation(state, normalized)
        if pending_flow == "smart_help_details":
            return await self._handle_client_smart_help(state, last_message)

        if last_message.isdigit() and last_message not in [str(i) for i in range(1, 9)]:
            state["context"]["intent"] = "invalid_client_menu_option"
            state["context"]["tool_response"] = "Please reply with a number from 1 to 8. Reply 9 for a human agent or 10 for the main menu."
            return state

        if last_message == "1":
            state["context"]["intent"] = "schedule_pickup"
            state["context"]["client_flow"] = "pickup_choice"
            addresses = fetch_client_addresses(mobile)
            state["context"]["saved_addresses"] = addresses
            if addresses:
                addr_lines = "\n".join(
                    format_address_label(addr, index=None)
                    for addr in addresses
                )
                state["context"]["tool_response"] = (
                    f"Your saved pickup address(es):\n{addr_lines}\n\n"
                    f"{CLIENT_PICKUP_CHOICE_PROMPT}"
                )
            else:
                state["context"]["tool_response"] = (
                    "No saved address found on your account.\n\n"
                    "Please provide your pickup address in the format:\n"
                    + CLIENT_NEW_ADDRESS_PROMPT
                )
                state["context"]["client_flow"] = "awaiting_new_address"
            return state

        if last_message == "2":
            state["context"]["intent"] = "price_catalogue"
            state["context"]["client_flow"] = "pricing_followup"
            state["context"]["tool_response"] = (
                f"Our price catalogue is:\n\n{fetch_items_with_price()}\n\n"
                "Would you like to:\n"
                "1. Schedule fresh pick up\n"
                "2. Will reach out later"
            )
            return state

        if last_message == "3":
            state["context"]["intent"] = "visit_appointment"
            state["context"]["client_flow"] = "store_visit_date_selection"
            history = fetch_client_appointment_history(mobile)
            latest_note = ""
            if history:
                latest = history[0]
                last_date = latest.get("bookdate") or latest.get("date") or ""
                last_time = latest.get("booktime") or latest.get("time") or ""
                if last_date or last_time:
                    latest_note = f"Your last appointment: {last_date} {last_time}\n\n"
            state["context"]["tool_response"] = latest_note + CLIENT_VISIT_DATE_PROMPT
            return state

        if last_message == "4":
            state["context"]["intent"] = "estimate_fabric_requirement"
            state["context"]["client_flow"] = "estimate_details"
            state["context"]["tool_response"] = CLIENT_ESTIMATE_PROMPT
            return state

        if last_message == "5":
            return self._handle_client_measurements_menu(state, mobile)

        if last_message == "6":
            state["context"]["intent"] = "store_address"
            state["context"]["client_flow"] = None
            client_info = fetch_customer_info_data(mobile)
            city = client_info.get("city") or "Hyderabad"
            state["context"]["tool_response"] = f"Our store address for {city} is:\n\n{STORE_VISIT_INFO}"
            return state

        if last_message == "7":
            state["context"]["intent"] = "fabric_delivery_to_store"
            state["context"]["client_flow"] = None
            client_info = fetch_customer_info_data(mobile)
            customer_name = (
                state["context"].get("customer_name")
                or client_info.get("cname")
                or "Customer"
            )
            customer_segment = state["context"].get("customer_segment", "client")
            label = generate_store_delivery_label_code(
                customer_id=state.get("conversation_id", mobile),
                customer_name=customer_name,
                customer_segment=customer_segment,
                mobile=mobile,
            )
            state["context"]["tool_response"] = (
                "Please use the below delivery label while sending fabric to our store via Rapido or Uber:\n\n"
                f"Delivery Label Code: {label['label_code']}\n"
                f"Customer Name: {customer_name}\n"
                f"Customer Mobile: {mobile}\n\n"
                "Store Drop Address:\n"
                f"{STORE_DELIVERY_ADDRESS}\n\n"
                f"Store Contact Number: {STORE_DELIVERY_PHONE}\n\n"
                "Important: Please ask the delivery partner to mention the label code at drop-off."
            )
            return state

        if last_message == "8":
            state["context"]["intent"] = "client_smart_help_start"
            state["context"]["client_flow"] = "smart_help_details"
            state["context"]["tool_response"] = CLIENT_SMART_HELP_PROMPT
            return state

        return None

    async def _handle_client_pickup_choice(
        self,
        state: AgentState,
        last_message: str,
        normalized: str,
        mobile: str,
    ) -> AgentState:
        # Option 1: schedule from saved address
        if normalized in {"1", "1a", "a", "savedaddress", "pickupfromsavedaddress", "schedulepickupfromsavedaddress"}:
            addresses = state["context"].get("saved_addresses") or fetch_client_addresses(mobile)
            if not addresses:
                # No saved addresses, fall through to new address entry
                state["context"]["intent"] = "update_pickup_address"
                state["context"]["client_flow"] = "awaiting_new_address"
                state["context"]["tool_response"] = (
                    "No saved address found. Please provide your pickup address:\n\n"
                    + CLIENT_NEW_ADDRESS_PROMPT
                )
                return state

            if len(addresses) == 1:
                # Only one address — skip selection, ask for date then time slot
                addr = addresses[0]
                label = format_address_label(addr)
                state["context"]["intent"] = "pickup_date_selection"
                state["context"]["client_flow"] = "pickup_date_selection"
                state["context"]["selected_address"] = addr
                state["context"]["tool_response"] = (
                    f"Your pickup address:\n{label}\n\n"
                    + CLIENT_PICKUP_DATE_PROMPT
                )
                return state

            # Multiple addresses — ask customer to pick one
            options = "\n".join(
                format_address_label(addr, index=i + 1)
                for i, addr in enumerate(addresses)
            )
            state["context"]["intent"] = "select_saved_address"
            state["context"]["client_flow"] = "select_saved_address"
            state["context"]["saved_addresses"] = addresses
            state["context"]["tool_response"] = (
                f"Please select the address for pickup:\n\n{options}\n\n"
                "Reply with the number of your preferred address."
            )
            return state

        # Option 2: add new address
        if normalized in {"2", "1b", "b", "newaddress", "addaddress", "addnewaddress"}:
            state["context"]["intent"] = "add_new_pickup_address"
            state["context"]["client_flow"] = "awaiting_new_address"
            state["context"]["tool_response"] = CLIENT_NEW_ADDRESS_PROMPT
            return state

        state["context"]["intent"] = "pickup_choice_clarification"
        state["context"]["tool_response"] = "Please reply with 1 to schedule pickup from your saved address or 2 to add a new address."
        return state

    async def _handle_client_select_saved_address(
        self,
        state: AgentState,
        last_message: str,
        normalized: str,
        mobile: str,
    ) -> AgentState:
        addresses = state["context"].get("saved_addresses") or []
        if not normalized.isdigit():
            state["context"]["intent"] = "select_address_clarification"
            state["context"]["tool_response"] = "Please reply with the number of the address you want to use for pickup."
            return state

        index = int(normalized) - 1
        if index < 0 or index >= len(addresses):
            state["context"]["intent"] = "select_address_clarification"
            state["context"]["tool_response"] = (
                f"Please reply with a number between 1 and {len(addresses)} to select your pickup address."
            )
            return state

        addr = addresses[index]
        label = format_address_label(addr)
        state["context"]["intent"] = "pickup_date_selection"
        state["context"]["client_flow"] = "pickup_date_selection"
        state["context"]["selected_address"] = addr
        state["context"]["saved_addresses"] = None
        state["context"]["tool_response"] = (
            f"Your pickup address:\n{label}\n\n"
            + CLIENT_PICKUP_DATE_PROMPT
        )
        logger.info(f"Address selected index={index+1} for mobile={mobile}: {label}, now asking pickup date")
        return state

    async def _handle_client_new_address(self, state: AgentState, last_message: str, mobile: str) -> AgentState:
        parsed_address = await self._extract_client_address(last_message, mobile)
        if not parsed_address.get("is_complete"):
            missing = parsed_address.get("missing_fields") or []
            state["context"]["intent"] = "address_details_incomplete"
            state["context"]["tool_response"] = self._build_address_retry_message(missing)
            return state

        sender_digits = "".join(char for char in str(state.get("conversation_id", "")) if char.isdigit())
        crm_mobile_input = sender_digits or mobile
        
        # Try to get existing address_id for update action
        existing_client = fetch_customer_info_data(mobile)
        address_id = existing_client.get("address_id")
        
        logger.info(f"Address update for mobile={mobile}: address_id={address_id}, parsed_address={parsed_address}")
        
        # Use update if address_id exists, otherwise use add
        if address_id:
            result = update_client_address(crm_mobile_input, address_id, parsed_address)
            logger.info(f"Called update_client_address: result={result}")
        else:
            result = add_client_address(crm_mobile_input, parsed_address)
            logger.info(f"Called add_client_address: result={result}")
        
        if not result.get("success"):
            state["context"]["intent"] = "address_update_failed"
            state["context"]["tool_response"] = (
                "I could not save that address right now. Please resend the full address with pincode, city, and geo coordinates."
            )
            return state

        refreshed_client = fetch_customer_info_data(mobile)
        refreshed_addresses = fetch_client_addresses(mobile)
        logger.info(f"Refreshed client data: {refreshed_client}")
        logger.info(f"Refreshed address list count: {len(refreshed_addresses)}")

        parsed_address1 = str(parsed_address.get("address1", "")).strip().lower()
        parsed_city = str(parsed_address.get("city", "")).strip().lower()
        parsed_pincode = str(parsed_address.get("pincode", "")).strip().lower()

        verification_candidates = []
        if isinstance(refreshed_client, dict) and refreshed_client:
            verification_candidates.append(refreshed_client)
        if isinstance(refreshed_addresses, list):
            verification_candidates.extend(
                addr for addr in refreshed_addresses if isinstance(addr, dict)
            )

        verified = False
        for candidate in verification_candidates:
            candidate_address1 = str(candidate.get("address1", "")).strip().lower()
            candidate_city = str(candidate.get("city", "")).strip().lower()
            candidate_pincode = str(candidate.get("pincode", "")).strip().lower()

            address_match = bool(parsed_address1 and parsed_address1 in candidate_address1)
            city_match = bool(parsed_city and parsed_city in candidate_city)
            pincode_match = bool(parsed_pincode and parsed_pincode in candidate_pincode)

            match_score = sum([address_match, city_match, pincode_match])
            if address_match or pincode_match or match_score >= 2:
                verified = True
                break

        if not verified:
            # Do not block pickup flow if CRM already acknowledged a successful save/update.
            logger.warning(
                "Address verification inconclusive after successful save. Proceeding to pickup scheduling. "
                "Parsed=%s",
                {"address1": parsed_address1, "city": parsed_city, "pincode": parsed_pincode},
            )

        state["context"]["intent"] = "pickup_date_selection"
        state["context"]["client_flow"] = "pickup_date_selection"
        saved_address = ", ".join(
            value for value in [
                result["payload"].get("address1"),
                result["payload"].get("address2"),
                result["payload"].get("locality"),
                result["payload"].get("city"),
                result["payload"].get("pincode"),
            ]
            if value
        )
        state["context"]["selected_address_label"] = saved_address
        state["context"]["tool_response"] = (
            f"✓ Your pickup address has been saved:\n{saved_address}\n\n"
            + CLIENT_PICKUP_DATE_PROMPT
        )
        logger.info(f"Address update verified for mobile={mobile}, asking for pickup date")
        return state

    async def _handle_client_pickup_date_choice(
        self,
        state: AgentState,
        last_message: str,
        normalized: str,
    ) -> AgentState:
        from datetime import date as _date, datetime as _datetime, timedelta as _timedelta

        if normalized in {"1", "today"}:
            pickup_date = _date.today().isoformat()
        elif normalized in {"2", "tomorrow"}:
            pickup_date = (_date.today() + _timedelta(days=1)).isoformat()
        elif normalized in {"3", "custom", "other", "anotherdate"}:
            state["context"]["intent"] = "pickup_date_custom_entry"
            state["context"]["tool_response"] = "Please enter your pickup date in YYYY-MM-DD format."
            return state
        else:
            date_text = last_message.strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
                state["context"]["intent"] = "pickup_date_clarification"
                state["context"]["tool_response"] = (
                    "Please reply with:\n"
                    "1. Today\n"
                    "2. Tomorrow\n"
                    "3. Enter another date (YYYY-MM-DD)"
                )
                return state
            try:
                parsed_date = _datetime.strptime(date_text, "%Y-%m-%d").date()
            except ValueError:
                state["context"]["intent"] = "pickup_date_invalid"
                state["context"]["tool_response"] = "Invalid date format. Please enter a valid date as YYYY-MM-DD."
                return state
            if parsed_date < _date.today():
                state["context"]["intent"] = "pickup_date_past"
                state["context"]["tool_response"] = "Pickup date cannot be in the past. Please choose today, tomorrow, or a future date."
                return state
            pickup_date = parsed_date.isoformat()

        state["context"]["selected_pickup_date"] = pickup_date
        state["context"]["intent"] = "pickup_time_slot_selection"
        state["context"]["client_flow"] = "pickup_time_slot"
        state["context"]["tool_response"] = (
            f"Selected pickup date: {pickup_date}\n\n"
            + CLIENT_PICKUP_TIME_PROMPT
        )
        return state

    async def _handle_client_pickup_time_slot(
        self,
        state: AgentState,
        last_message: str,
        normalized: str,
        mobile: str,
    ) -> AgentState:
        slot = PICKUP_TIME_SLOTS.get(normalized)
        if not slot:
            state["context"]["intent"] = "pickup_time_slot_clarification"
            state["context"]["tool_response"] = (
                "Please reply with:\n"
                "1. Morning (9AM \u2013 2PM)\n"
                "2. Afternoon (2PM \u2013 9PM)"
            )
            return state

        # Determine the address to use
        selected_address = state["context"].get("selected_address") or {}
        address_label = (
            state["context"].get("selected_address_label")
            or format_address_label(selected_address)
            if selected_address
            else state["context"].get("selected_address_label", "your saved address")
        )

        pickup_date = str(state["context"].get("selected_pickup_date") or "")
        if not pickup_date:
            from datetime import date as _date
            pickup_date = str(_date.today())

        result = schedule_pickup_crm(mobile, pickup_date, slot["value"])
        logger.info(f"schedule_pickup_crm result for mobile={mobile}: {result}")

        state["context"]["client_flow"] = None
        state["context"]["selected_address"] = None
        state["context"]["selected_address_label"] = None
        state["context"]["selected_pickup_date"] = None

        if result.get("success"):
            state["context"]["intent"] = "pickup_scheduled"
            state["context"]["tool_response"] = (
                f"\u2713 Pickup scheduled!\n\n"
                f"Address: {address_label}\n"
                f"Date: {pickup_date}\n"
                f"Time slot: {slot['label']}\n\n"
                "Our team will confirm your pickup shortly. Thank you!"
            )
        else:
            state["context"]["intent"] = "pickup_schedule_failed"
            failure_reason = str(result.get("message") or "Could not confirm pickup scheduling.").strip()
            state["context"]["tool_response"] = (
                f"I could not confirm the pickup booking right now.\n\n"
                f"Reason: {failure_reason}\n"
                f"Address: {address_label}\n"
                f"Date: {pickup_date}\n"
                f"Time slot: {slot['label']}\n\n"
                "Please reply 9 and our team will schedule it manually for you."
            )
        return state

    async def _handle_client_store_visit_date_choice(
        self,
        state: AgentState,
        last_message: str,
        normalized: str,
        mobile: str,
    ) -> AgentState:
        from datetime import date as _date, datetime as _datetime, timedelta as _timedelta

        if normalized in {"1", "today"}:
            visit_date = _date.today().isoformat()
        elif normalized in {"2", "tomorrow"}:
            visit_date = (_date.today() + _timedelta(days=1)).isoformat()
        elif normalized in {"3", "custom", "other", "anotherdate"}:
            state["context"]["intent"] = "store_visit_custom_date_entry"
            state["context"]["tool_response"] = "Please enter your visit date in YYYY-MM-DD format."
            return state
        else:
            date_text = last_message.strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
                state["context"]["intent"] = "store_visit_date_clarification"
                state["context"]["tool_response"] = CLIENT_VISIT_DATE_PROMPT
                return state
            try:
                parsed_date = _datetime.strptime(date_text, "%Y-%m-%d").date()
            except ValueError:
                state["context"]["intent"] = "store_visit_date_invalid"
                state["context"]["tool_response"] = "Invalid date format. Please enter a valid date as YYYY-MM-DD."
                return state
            if parsed_date < _date.today():
                state["context"]["intent"] = "store_visit_date_past"
                state["context"]["tool_response"] = "Visit date cannot be in the past. Please choose today, tomorrow, or a future date."
                return state
            visit_date = parsed_date.isoformat()

        state["context"]["selected_visit_date"] = visit_date

        # Fetch exact available slots right after date selection.
        # CRM returns available_slots when a broad range is invalid.
        probe = book_store_appointment(
            mobile=mobile,
            bookdate=visit_date,
            booktime="11 AM - 2PM",
            store_id=1,
        )
        available_slots = []
        if isinstance(probe.get("response"), dict):
            slots = probe["response"].get("available_slots")
            if isinstance(slots, list):
                available_slots = [str(slot).strip() for slot in slots if str(slot).strip()]

        if available_slots:
            options = "\n".join(
                f"{idx}. {slot}"
                for idx, slot in enumerate(available_slots, start=1)
            )
            state["context"]["intent"] = "store_visit_exact_slot_selection"
            state["context"]["client_flow"] = "store_visit_exact_slot_selection"
            state["context"]["available_visit_slots"] = available_slots
            state["context"]["tool_response"] = (
                "The selected slot range is not available. Please choose an exact available slot:\n\n"
                f"Date: {visit_date}\n"
                f"{options}\n\n"
                "Reply with the slot number."
            )
            return state

        failure_reason = str(probe.get("message") or "Could not fetch available appointment slots.").strip()
        state["context"]["intent"] = "store_visit_slot_fetch_failed"
        state["context"]["client_flow"] = None
        state["context"]["tool_response"] = (
            "I could not fetch available store visit slots right now.\n\n"
            f"Reason: {failure_reason}\n"
            f"Date: {visit_date}\n\n"
            "Please reply 9 and our team will book your appointment manually."
        )
        return state

    async def _handle_client_store_visit_exact_slot_choice(
        self,
        state: AgentState,
        normalized: str,
        mobile: str,
    ) -> AgentState:
        slots = state["context"].get("available_visit_slots") or []
        if not normalized.isdigit():
            state["context"]["intent"] = "store_visit_exact_slot_clarification"
            state["context"]["tool_response"] = "Please reply with the slot number from the list."
            return state

        index = int(normalized) - 1
        if index < 0 or index >= len(slots):
            state["context"]["intent"] = "store_visit_exact_slot_clarification"
            state["context"]["tool_response"] = f"Please reply with a number between 1 and {len(slots)}."
            return state

        exact_slot = slots[index]
        visit_date = str(state["context"].get("selected_visit_date") or "")

        result = book_store_appointment(
            mobile=mobile,
            bookdate=visit_date,
            booktime=exact_slot,
            store_id=1,
        )
        logger.info(f"book_store_appointment exact-slot result for mobile={mobile}: {result}")

        state["context"]["client_flow"] = None
        state["context"]["selected_visit_date"] = None
        state["context"]["available_visit_slots"] = None

        if result.get("success"):
            state["context"]["intent"] = "store_visit_booked"
            state["context"]["tool_response"] = (
                "✓ Your store visit appointment is booked.\n\n"
                f"Date: {visit_date}\n"
                f"Time: {exact_slot}\n"
                "Store: Jubilee Hills, Hyderabad\n\n"
                "We look forward to seeing you."
            )
        else:
            failure_reason = str(result.get("message") or "Could not confirm appointment booking.").strip()
            state["context"]["intent"] = "store_visit_booking_failed"
            state["context"]["tool_response"] = (
                "I could not confirm your store appointment right now.\n\n"
                f"Reason: {failure_reason}\n"
                f"Date: {visit_date}\n"
                f"Time: {exact_slot}\n\n"
                "Please reply 9 and our team will book it manually for you."
            )
        return state

    def _handle_client_pricing_followup(self, state: AgentState, normalized: str) -> AgentState:
        if normalized in {"1", "schedulefreshpickup", "schedulepickup", "pickup"}:
            state["context"]["intent"] = "schedule_pickup"
            state["context"]["client_flow"] = "pickup_choice"
            mobile = normalize_mobile(state.get("conversation_id", ""))
            addresses = fetch_client_addresses(mobile)
            state["context"]["saved_addresses"] = addresses
            if addresses:
                addr_lines = "\n".join(format_address_label(addr) for addr in addresses)
                state["context"]["tool_response"] = (
                    f"Your saved pickup address(es):\n{addr_lines}\n\n"
                    f"{CLIENT_PICKUP_CHOICE_PROMPT}"
                )
            else:
                state["context"]["tool_response"] = (
                    "No saved address found. Please provide your pickup address:\n\n"
                    + CLIENT_NEW_ADDRESS_PROMPT
                )
                state["context"]["client_flow"] = "awaiting_new_address"
            return state

        if normalized in {"2", "willreachoutlater", "later", "reachoutlater", "no"}:
            state["context"]["intent"] = "price_catalogue_complete"
            state["context"]["client_flow"] = None
            state["context"]["tool_response"] = CLIENT_LATER_RESPONSE
            return state

        state["context"]["intent"] = "price_catalogue_followup_clarification"
        state["context"]["tool_response"] = "Please reply with 1 to schedule a fresh pick up or 2 if you will reach out later."
        return state

    def _handle_client_measurements_menu(self, state: AgentState, mobile: str) -> AgentState:
        measurement_forms = fetch_customer_measurement_forms(mobile)
        if not measurement_forms:
            state["context"]["intent"] = "measurements"
            state["context"]["client_flow"] = None
            state["context"]["tool_response"] = fetch_customer_measurements(mobile)
            return state

        options_text = "\n".join(
            f"{index}. {form['label']}"
            for index, form in enumerate(measurement_forms, start=1)
        )
        state["context"]["intent"] = "measurement_options"
        state["context"]["client_flow"] = "measurement_choice"
        state["context"]["measurement_options"] = measurement_forms
        state["context"]["tool_response"] = f"Would you like us to display measurements for:\n{options_text}"
        return state

    def _handle_client_measurement_choice(self, state: AgentState, normalized: str) -> AgentState:
        measurement_forms = state["context"].get("measurement_options") or []
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(measurement_forms):
                selected_form = measurement_forms[index]
                state["context"]["intent"] = "measurements"
                state["context"]["client_flow"] = None
                state["context"]["measurement_options"] = None
                state["context"]["tool_response"] = selected_form["details"]
                return state

        state["context"]["intent"] = "measurement_choice_clarification"
        state["context"]["tool_response"] = "Please reply with the number of the measurement profile you want to view."
        return state

    def _handle_client_shipping_label_confirmation(self, state: AgentState, normalized: str) -> AgentState:
        if normalized in {"yes", "y", "generate", "generateshippinglabel", "ok", "sure"}:
            state["context"]["intent"] = "shipping_label_requested"
            state["context"]["client_flow"] = None
            state["context"]["tool_response"] = "We will arrange the shipping label and share it with you shortly."
            state["context"]["handoff_after_response"] = True
            return state

        if normalized in {"no", "n", "later"}:
            state["context"]["intent"] = "shipping_label_declined"
            state["context"]["client_flow"] = None
            state["context"]["tool_response"] = "No problem. When you are ready, message us and we will arrange store delivery support."
            return state

        state["context"]["intent"] = "shipping_label_clarification"
        state["context"]["tool_response"] = "Please reply yes if you want the shipping label, or no if you want to skip it for now."
        return state

    async def _handle_client_smart_help(self, state: AgentState, message: str) -> AgentState:
        prompt = (
            "Classify this tailoring support issue and suggest the next step. "
            "Return JSON with keys: category, needs_human, customer_reply. "
            "Valid categories: order_delay, payment_issue, alteration_issue, pickup_issue, delivery_issue, quality_issue, billing_issue, other. "
            "Set needs_human=true when issue is urgent, complaint-like, unresolved, or requires manual intervention.\n\n"
            f"Message: {json.dumps(message)}\n"
            "Output example: {\"category\": \"delivery_issue\", \"needs_human\": true, \"customer_reply\": \"I understand. I will connect you to our support team now.\"}"
        )

        response = await self.llm.ainvoke(prompt)
        parsed = self._safe_parse_json(response.content)
        raw_category = str(parsed.get("category", "")).strip().lower()
        category = raw_category if raw_category in HELP_CATEGORY_TEMPLATES else self._infer_help_category_from_text(message)

        parsed_needs_human = parsed.get("needs_human")
        if isinstance(parsed_needs_human, bool):
            needs_human = parsed_needs_human
        else:
            needs_human = category in HELP_CATEGORY_AUTO_ESCALATE

        if any(token in message.lower() for token in ["human", "agent", "representative", "support"]):
            needs_human = True

        customer_reply = str(parsed.get("customer_reply", "")).strip()

        if not customer_reply:
            customer_reply = HELP_CATEGORY_TEMPLATES.get(category, HELP_CATEGORY_TEMPLATES["other"])

        state["context"]["intent"] = f"client_smart_help_{category}"
        state["context"]["client_flow"] = None
        state["context"]["help_category"] = category
        state["context"]["tool_response"] = customer_reply

        if needs_human:
            state["context"]["handoff_after_response"] = True

        return state

    def _infer_help_category_from_text(self, message: str) -> str:
        text = message.lower()
        keyword_map = {
            "order_delay": ["delay", "late", "not ready", "status", "stuck"],
            "payment_issue": ["payment", "paid", "upi", "refund", "transaction"],
            "alteration_issue": ["alter", "alteration", "fit", "tight", "loose", "size"],
            "pickup_issue": ["pickup", "collect", "reschedule", "not picked"],
            "delivery_issue": ["delivery", "courier", "dispatch", "not delivered"],
            "quality_issue": ["quality", "stitch", "defect", "damage", "poor"],
            "billing_issue": ["bill", "billing", "invoice", "charge", "tax"],
        }

        for category, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                return category
        return "other"

    async def _extract_client_address(self, message: str, mobile: str) -> dict:
        deterministic_payload, deterministic_match = self._parse_address_from_structured_text(message, mobile)
        if deterministic_match:
            return deterministic_payload

        prompt = (
            "Extract a pickup address from this customer message for a tailoring CRM. "
            "Return valid JSON with keys: address1, address2, locality, city, pincode, lat, lng, tel, is_complete, missing_fields. "
            "Use empty strings for missing values and a JSON array for missing_fields. "
            f"Default tel should be {mobile}.\n\n"
            f"Message: {json.dumps(message)}\n"
            "Output example: {\"address1\": \"12-3 MG Road\", \"address2\": \"Near Park\", \"locality\": \"Banjara Hills\", \"city\": \"Hyderabad\", \"pincode\": \"500034\", \"lat\": \"17.4301\", \"lng\": \"78.4230\", \"tel\": \"9908712226\", \"is_complete\": true, \"missing_fields\": []}"
        )

        response = await self.llm.ainvoke(prompt)
        parsed = self._safe_parse_json(response.content)
        payload = {
            "address1": str(parsed.get("address1", "")).strip(),
            "address2": str(parsed.get("address2", "")).strip(),
            "locality": str(parsed.get("locality", "")).strip(),
            "city": str(parsed.get("city", "")).strip(),
            "pincode": str(parsed.get("pincode", "")).strip(),
            "lat": str(parsed.get("lat", "")).strip(),
            "lng": str(parsed.get("lng", "")).strip(),
            "tel": str(parsed.get("tel") or mobile).strip(),
        }
        payload = self._validate_address_payload(payload)
        return payload

    def _parse_address_from_structured_text(self, message: str, mobile: str) -> tuple[dict, bool]:
        lines = [line.strip().rstrip(",") for line in message.splitlines() if line.strip()]
        payload = {
            "address1": "",
            "address2": "",
            "locality": "",
            "city": "",
            "pincode": "",
            "lat": "",
            "lng": "",
            "tel": str(mobile).strip(),
        }

        labels = {
            "address line 1": "address1",
            "address line 2": "address2",
            "locality": "locality",
            "city": "city",
            "pincode": "pincode",
            "latitude": "lat",
            "longitude": "lng",
        }

        labeled_hits = 0
        for line in lines:
            if ":" not in line:
                continue
            key_text, value = line.split(":", 1)
            key = key_text.strip().lower()
            for label, field in labels.items():
                if key.startswith(label):
                    payload[field] = value.strip()
                    labeled_hits += 1
                    break

        if labeled_hits >= 5:
            return self._validate_address_payload(payload), True

        if len(lines) == 7:
            payload.update({
                "address1": lines[0],
                "address2": lines[1],
                "locality": lines[2],
                "city": lines[3],
                "pincode": lines[4],
                "lat": lines[5],
                "lng": lines[6],
            })
            return self._validate_address_payload(payload), True

        comma_parts = [part.strip() for part in message.split(",") if part.strip()]
        if len(comma_parts) >= 7:
            payload.update({
                "address1": ", ".join(comma_parts[:-6]),
                "address2": comma_parts[-6],
                "locality": comma_parts[-5],
                "city": comma_parts[-4],
                "pincode": comma_parts[-3],
                "lat": comma_parts[-2],
                "lng": comma_parts[-1],
            })
            return self._validate_address_payload(payload), True

        return self._validate_address_payload(payload), False

    def _validate_address_payload(self, payload: dict) -> dict:
        required_fields = ["address1", "locality", "city", "pincode", "lat", "lng"]
        missing_fields = [field for field in required_fields if not str(payload.get(field, "")).strip()]

        pincode = str(payload.get("pincode", "")).strip()
        if pincode and not re.fullmatch(r"\d{6}", pincode):
            missing_fields.append("valid_6_digit_pincode")

        lat = str(payload.get("lat", "")).strip()
        lng = str(payload.get("lng", "")).strip()
        if lat:
            try:
                lat_value = float(lat)
                if lat_value < -90 or lat_value > 90:
                    missing_fields.append("valid_latitude")
            except ValueError:
                missing_fields.append("valid_latitude")

        if lng:
            try:
                lng_value = float(lng)
                if lng_value < -180 or lng_value > 180:
                    missing_fields.append("valid_longitude")
            except ValueError:
                missing_fields.append("valid_longitude")

        payload["missing_fields"] = list(dict.fromkeys(missing_fields))
        payload["is_complete"] = not payload["missing_fields"]
        return payload

    def _build_address_retry_message(self, missing_fields: list[str]) -> str:
        if not missing_fields:
            missing_fields = ["city", "valid_6_digit_pincode", "valid_latitude", "valid_longitude"]

        guidance_map = {
            "address1": "Line 1 (Address line 1) is missing.",
            "address2": "Line 2 (Address line 2 / nearest landmark) is missing.",
            "locality": "Line 3 (Locality) is missing.",
            "city": "Line 4 (City) is missing.",
            "pincode": "Line 5 (Pincode) is missing.",
            "lat": "Line 6 (Latitude) is missing.",
            "lng": "Line 7 (Longitude) is missing.",
            "valid_6_digit_pincode": "Line 5 must be a valid 6-digit pincode.",
            "valid_latitude": "Line 6 must be a valid latitude value between -90 and 90.",
            "valid_longitude": "Line 7 must be a valid longitude value between -180 and 180.",
        }

        details = [guidance_map[field] for field in missing_fields if field in guidance_map]
        details_text = "\n".join(f"- {item}" for item in details) if details else "- Please resend all 7 lines in the required format."

        return (
            "I could not validate the pickup address yet. Please correct these lines and resend:\n"
            f"{details_text}\n\n"
            "Required format:\n"
            "Address line 1: <value>\n"
            "Address line 2 (nearest landmark): <value>\n"
            "Locality: <value>\n"
            "City: <value>\n"
            "Pincode: <6-digit value>\n"
            "Latitude: <value>\n"
            "Longitude: <value>"
        )

    async def generate_response(self, state: AgentState) -> AgentState:
        if state["needs_human"]:
            response_text = (
                "I’m connecting you with a Tailorsin specialist now. "
                "Please hold on while I hand this over to our human support team."
            )
        else:
            tool_reply = state["context"].get("tool_response")
            intent = state["context"].get("intent", "general")

            if tool_reply:
                # Customize response based on intent
                if intent in ("order_status", "order_ready_time", "tracking", "order_history"):
                    response_text = f"📦 {tool_reply}"
                elif intent in ("pricing", "add_items", "new_order"):
                    response_text = f"🧵 {tool_reply}"
                elif intent in ("measurements", "update_measurements", "measurement_process"):
                    response_text = f"📏 {tool_reply}"
                elif intent in ("browse_items", "browse_designs", "fabrics"):
                    response_text = f"🛍️ {tool_reply}"
                elif intent == "whatsapp_history":
                    response_text = f"💬 {tool_reply}"
                elif intent == "pickup_delivery":
                    response_text = f"🚚 {tool_reply}"
                else:
                    response_text = str(tool_reply)
            else:
                response_text = await self._get_llm_response(state)

        nav_footer = "\n\n─────────────────────\nReply 9 for human agent | 10 for the main menu\nYou can also type menu or human anytime."
        if not state["needs_human"] and not state["context"].get("handoff_after_response"):
            response_text = response_text + nav_footer

        state["messages"].append({"role": "assistant", "content": response_text})
        return state

    async def _get_llm_response(self, state: AgentState) -> str:
        context_messages = state["messages"][-6:]
        context_text = "\n".join([f"{m['role']}: {m['content']}" for m in context_messages])

        prompt = (
            "You are the WhatsApp concierge for Tailorsin.com, a premium bespoke tailoring service. "
            "You help customers with order updates, garments, pricing, pickup, delivery, measurements, store visits, and onboarding. "
            "Answer in a polished, warm, operational tone. Be concise, clear, and helpful. "
            "When possible, give the next step the customer should take. Avoid sounding robotic or generic.\n\n"
            "Conversation so far:\n" + context_text + "\n\n"
            "Answer the customer's last question directly."
        )

        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    def _safe_parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to recover JSON from text if the model emitted extra explanation
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        return {}
