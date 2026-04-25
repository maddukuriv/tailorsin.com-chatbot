from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from app.agent import E_TailoringAgent
from app.storage import RedisConversationStore
from app.tools import classify_customer_profile, normalize_mobile

load_dotenv()

app = FastAPI(title="E-Tailoring WhatsApp Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
HUMAN_HANDOFF_WHATSAPP_NUMBER = os.getenv("HUMAN_HANDOFF_WHATSAPP_NUMBER")
SUPPORT_API_TOKEN = os.getenv("SUPPORT_API_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ALLOW_INVALID_TWILIO_SIGNATURE = os.getenv("ALLOW_INVALID_TWILIO_SIGNATURE", "false").strip().lower() in {"1", "true", "yes", "on"}
WEBHOOK_JSON_TEST_MODE = os.getenv("WEBHOOK_JSON_TEST_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

if LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required in .env when LLM_PROVIDER=groq")
    LLM_API_KEY = GROQ_API_KEY
    LLM_MODEL = GROQ_MODEL
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required in .env when LLM_PROVIDER=openai")
    LLM_API_KEY = OPENAI_API_KEY
    LLM_MODEL = OPENAI_MODEL
else:
    raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'openai'")

agent = E_TailoringAgent(provider=LLM_PROVIDER, api_key=LLM_API_KEY, model=LLM_MODEL)
store = RedisConversationStore(REDIS_URL)
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "10"))


GREETINGS = {"hi", "hello", "hey", "hii", "helo", "yo", "namaste", "start", "menu", "help"}
MENU_COMMANDS = {"0", "menu", "main menu", "show menu", "home", "options", "start", "help"}
HUMAN_COMMANDS = {"00", "human", "agent", "support", "customer care", "representative", "talk to human", "talk to agent"}


def normalize_command(text: str) -> str:
    return " ".join(text.lower().strip().rstrip("!.,?").split())

def is_greeting(text: str) -> bool:
    return normalize_command(text) in GREETINGS


def is_menu_request(text: str) -> bool:
    normalized = normalize_command(text)
    return normalized in MENU_COMMANDS or normalized in GREETINGS


def is_human_request(text: str) -> bool:
    return normalize_command(text) in HUMAN_COMMANDS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_session_expired(conversation: dict) -> bool:
    last_activity = conversation.get("last_activity_at")
    if not last_activity:
        return False
    return utc_now() - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def reset_conversation_state(conversation: dict) -> None:
    conversation["messages"] = []
    conversation["handoff_active"] = False
    conversation["customer_profile"] = None
    conversation["agent_context"] = {}
    conversation["handoff_summary"] = None
    conversation["handoff_requested_at"] = None
    conversation["handoff_assigned_to"] = None
    conversation["handoff_last_human_message_at"] = None
    conversation["audit_log"] = []


def add_audit_event(conversation: dict, action: str, actor: str, details: dict | None = None) -> None:
    conversation.setdefault("audit_log", []).append({
        "action": action,
        "actor": actor,
        "details": details or {},
        "timestamp": utc_now(),
    })


SUPPORT_TOKEN_PLACEHOLDER = "replace-with-a-secure-random-token"

def ensure_support_access(x_support_token: str | None) -> None:
    configured = SUPPORT_API_TOKEN and SUPPORT_API_TOKEN != SUPPORT_TOKEN_PLACEHOLDER
    if configured and x_support_token != SUPPORT_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid support token")


def send_whatsapp_message(to_number: str, body: str) -> bool:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_NUMBER):
        logger.warning("Twilio outbound messaging is not fully configured.")
        return False

    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number,
        )
        return True
    except Exception as exc:
        logger.warning(f"Failed to send WhatsApp message to {to_number}: {exc}")
        return False


def twiml_message_response(message: str | None = None) -> PlainTextResponse:
    twilio_resp = MessagingResponse()
    if message is not None:
        twilio_resp.message(message)
    return PlainTextResponse(content=str(twilio_resp), media_type="text/xml")


class HandoffAssignRequest(BaseModel):
    agent_name: str


class HumanReplyRequest(BaseModel):
    message: str
    agent_name: str | None = None


class ResumeBotRequest(BaseModel):
    agent_name: str | None = None
    note: str | None = None

def build_segment_menu(profile: dict, include_greeting: bool = True) -> str:
    customer_name = profile.get("customer_name") or "there"
    segment = profile.get("segment", "new_user")
    customer_salutation = f"Mr/Ms {customer_name}" if customer_name != "there" else "there"
    footer = (
        "\n\n───────────\n"
        "9. Chat with a human agent\n"
        "10. Go back to the main menu\n\n"
        "Reply with a number from 1 to 10."
    )

    if segment == "active_client":
        greeting = f"Hello 👋\nWelcome back, {customer_salutation}, to tailorsin.com!\nI'm your AI assistant. How may we assist you today?\n\n" if include_greeting else ""
        return (
            greeting
            + "1. Check the status of my current order\n"
            "2. Check the expected readiness or delivery time\n"
            "3. Request changes to my current order\n"
            "4. View tracking or the latest delivery update\n"
            "5. View my saved measurements\n"
            "6. View my order history\n"
            "7. Add another item to my order\n"
            "8. View the price catalogue for additional items"
            + footer
        )

    if segment in ("client", "lead"):
        greeting = f"Hello 👋\nWelcome back, {customer_salutation}, to tailorsin.com!\nI'm your AI assistant. How may we assist you today?\n\n" if include_greeting else ""
        return (
            greeting
            + "1. Schedule a fresh pickup\n"
            "2. View the price catalogue\n"
            "3. Schedule a store visit\n"
            "4. Get a fabric requirement and price estimate\n"
            "5. View my saved measurements\n"
            "6. Get our store address\n"
            "7. Arrange fabric delivery to our store\n"
            "8. Get help with something else"
            + footer
        )

    # new_user — not in CRM
    greeting = ("Hi\n\n" + f"Hello {customer_salutation}, welcome to tailorsin.com. Here are the best options to learn about our service and get started.\n\n") if include_greeting else ""
    return (
        greeting
        + "1. Learn how tailorsin.com works\n"
        "2. Explore the garments we can stitch\n"
        "3. View the price catalogue\n"
        "4. Understand our measurement process\n"
        "5. Check delivery timelines\n"
        "6. Check our service areas\n"
        "7. Get the store address or schedule a visit\n"
        "8. Register and place your first order"
        + footer
    )

async def verify_twilio_request(request: Request, form_data: dict) -> bool:
    if not TWILIO_AUTH_TOKEN:
        return False  # No token = test request, return JSON

    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(TWILIO_AUTH_TOKEN)

    candidate_urls = []
    direct_url = str(request.url)
    candidate_urls.append(direct_url)

    forwarded_proto = request.headers.get("x-forwarded-proto")
    raw_path = request.url.path
    path_variants = [raw_path]
    if raw_path.endswith("/"):
        path_variants.append(raw_path.rstrip("/"))
    else:
        path_variants.append(f"{raw_path}/")

    query = request.url.query
    host_candidates = []
    for header in ("x-forwarded-host", "x-original-host", "host"):
        value = (request.headers.get(header) or "").strip()
        if value:
            host_candidates.append(value)

    normalized_hosts = []
    for host in host_candidates:
        normalized_hosts.append(host)
        if ":" in host:
            base_host, _, port = host.rpartition(":")
            if base_host and port in {"80", "443", "8000"}:
                normalized_hosts.append(base_host)

    schemes = []
    if forwarded_proto:
        schemes.append(forwarded_proto)
    schemes.extend(["https", "http"])

    for host in normalized_hosts:
        for scheme in schemes:
            for path in path_variants:
                candidate_url = f"{scheme}://{host}{path}"
                if query:
                    candidate_url = f"{candidate_url}?{query}"
                candidate_urls.append(candidate_url)

    seen_urls = set()
    for candidate_url in candidate_urls:
        if candidate_url in seen_urls:
            continue
        seen_urls.add(candidate_url)
        if validator.validate(candidate_url, form_data, signature):
            return True

    logger.warning(
        "Twilio signature validation failed for sender=%s url=%s forwarded_host=%s forwarded_proto=%s candidates=%s",
        form_data.get("From", ""),
        direct_url,
        request.headers.get("x-forwarded-host") or request.headers.get("host", ""),
        forwarded_proto or "",
        len(seen_urls),
    )
    return False

@app.get("/")
async def root():
    return {"status": "healthy", "service": "E-Tailoring WhatsApp Agent"}


@app.on_event("startup")
async def on_startup():
    try:
        await store.ping()
        logger.info("Redis connection verified.")
    except Exception as exc:
        logger.warning(
            f"Redis unavailable at startup ({exc}). "
            "The app will start but conversations won't be persisted until Redis is reachable."
        )


@app.get("/support", response_class=HTMLResponse)
async def support_dashboard():
    return """
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>Tailorsin Support Desk</title>
    <style>
        :root {
            --bg:#efe4d5;
            --panel:#fff9f1;
            --panel-strong:#fffdf9;
            --ink:#211a14;
            --muted:#6c6258;
            --line:#d9c8b4;
            --accent:#9a5a2c;
            --accent-2:#cf7f43;
            --ok:#256f4a;
            --warn:#8f5525;
        }
        * { box-sizing:border-box; }
        body {
            margin:0;
            font-family: Georgia, 'Times New Roman', serif;
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.7), transparent 28%),
                linear-gradient(180deg, #eadbc7, #f7f0e7 55%, #fbf8f2);
            color:var(--ink);
        }
        .wrap { max-width:1380px; margin:0 auto; padding:28px; }
        .top {
            display:grid;
            grid-template-columns: minmax(260px, 1fr) 240px 200px auto;
            gap:12px;
            align-items:center;
            margin-bottom:18px;
        }
        input, textarea, button, label { font:inherit; }
        input, textarea {
            border:1px solid var(--line);
            background:var(--panel-strong);
            padding:12px 14px;
            border-radius:12px;
            color:var(--ink);
        }
        button {
            border:none;
            background:linear-gradient(135deg, var(--accent), var(--accent-2));
            color:white;
            padding:12px 16px;
            border-radius:12px;
            cursor:pointer;
            font-weight:600;
        }
        button.secondary {
            background:#efe2d1;
            color:var(--ink);
            border:1px solid var(--line);
        }
        .layout {
            display:grid;
            grid-template-columns: 360px minmax(0, 1fr) 320px;
            gap:18px;
        }
        .card {
            background:rgba(255,249,241,0.96);
            border:1px solid var(--line);
            border-radius:20px;
            padding:18px;
            box-shadow:0 18px 42px rgba(43, 27, 14, 0.08);
            min-height:120px;
        }
        h1, h2, h3, p { margin-top:0; }
        h1 { font-size:28px; margin-bottom:6px; }
        h2 { font-size:18px; margin-bottom:12px; }
        h3 { font-size:15px; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); }
        .muted { color:var(--muted); font-size:14px; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        .pill {
            display:inline-flex;
            align-items:center;
            gap:6px;
            padding:6px 10px;
            border-radius:999px;
            border:1px solid var(--line);
            background:#fff;
            font-size:13px;
        }
        .statusbar {
            display:flex;
            justify-content:space-between;
            gap:10px;
            align-items:center;
            margin-bottom:12px;
            padding:10px 14px;
            border:1px solid var(--line);
            background:#fff;
            border-radius:14px;
        }
        .list { display:flex; flex-direction:column; gap:12px; max-height:72vh; overflow:auto; }
        .item {
            border:1px solid var(--line);
            border-radius:14px;
            padding:14px;
            cursor:pointer;
            background:#fffdf9;
            transition:transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
        }
        .item:hover { transform:translateY(-1px); border-color:var(--accent-2); box-shadow:0 10px 20px rgba(73, 41, 16, 0.08); }
        .item.active { border-color:var(--accent); box-shadow:0 12px 22px rgba(73, 41, 16, 0.12); }
        .item-head, .meta-row, .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
        .item-head { justify-content:space-between; margin-bottom:8px; }
        .conversation-head { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
        .summary-box, .audit-box, .transcript-box {
            border:1px solid var(--line);
            background:var(--panel-strong);
            border-radius:16px;
            padding:14px;
        }
        .summary-box { margin-bottom:14px; white-space:pre-wrap; line-height:1.45; }
        .messages { max-height:44vh; overflow:auto; display:flex; flex-direction:column; gap:10px; margin:0; }
        .msg {
            padding:12px 14px;
            border-radius:16px;
            border:1px solid var(--line);
            background:#fff;
        }
        .msg.user { border-left:5px solid #c48852; }
        .msg.assistant { border-left:5px solid #8f6fb5; }
        .msg.human { border-left:5px solid var(--ok); }
        .msg-role { font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-bottom:6px; }
        .msg-time, .audit-time { font-size:12px; color:var(--muted); }
        .msg-content { white-space:pre-wrap; line-height:1.45; }
        textarea { width:100%; min-height:120px; resize:vertical; }
        .audit-list { max-height:58vh; overflow:auto; display:flex; flex-direction:column; gap:10px; }
        .audit-item {
            border:1px solid var(--line);
            background:#fff;
            border-radius:14px;
            padding:12px;
        }
        .audit-action { font-weight:700; margin-bottom:4px; }
        .ok { color:var(--ok); }
        .warn { color:var(--warn); }
        .empty { padding:22px; border:1px dashed var(--line); border-radius:16px; text-align:center; color:var(--muted); }
        .toggle { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:14px; }
        .top-note { margin-bottom:18px; color:var(--muted); }
        .tabs { display:flex; gap:8px; margin-bottom:14px; }
        .tab {
            padding:8px 16px;
            border-radius:999px;
            border:1px solid var(--line);
            background:#fff;
            cursor:pointer;
            font-size:13px;
            font-weight:600;
            transition:background 120ms, border-color 120ms, color 120ms;
        }
        .tab:hover { border-color:var(--accent-2); }
        .tab.active { background:linear-gradient(135deg, var(--accent), var(--accent-2)); color:#fff; border-color:transparent; }
        .canned-section { margin-top:14px; }
        .canned-label { font-size:12px; text-transform:uppercase; letter-spacing:0.07em; color:var(--muted); margin-bottom:8px; }
        .canned-btns { display:flex; flex-wrap:wrap; gap:8px; }
        .canned-btn {
            padding:7px 13px;
            border-radius:10px;
            border:1px solid var(--line);
            background:#fffdf9;
            cursor:pointer;
            font-size:13px;
            transition:border-color 120ms, background 120ms;
        }
        .canned-btn:hover { border-color:var(--accent-2); background:#fff5ea; }
        @media (max-width: 1120px) {
            .layout { grid-template-columns: 1fr; }
            .top { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <h1>Tailorsin Support Desk</h1>
        <p class=\"top-note\">Monitor open escalations, reply as a human through the same WhatsApp thread, and resume the bot when the case is resolved.</p>
        <div class=\"top\">
            <input id=\"token\" placeholder=\"Support API token\" />
            <input id=\"agent\" placeholder=\"Agent name\" />
            <label class=\"toggle\"><input id=\"autoRefresh\" type=\"checkbox\" checked /> Auto-refresh every 10s</label>
            <button onclick=\"loadHandoffs()\">Refresh Queue</button>
        </div>
        <div class=\"statusbar\">
            <div class=\"meta-row\">
                <span class=\"pill\" id=\"queueCount\">0 open handoffs</span>
                <span class=\"pill\" id=\"lastRefresh\">Not refreshed yet</span>
            </div>
            <div id=\"flash\" class=\"muted\">Ready</div>
        </div>
        <div class=\"layout\">
            <div class=\"card\">
                <h2>Open Handoffs</h2>
                <div class=\"tabs\">
                    <button class=\"tab active\" data-filter=\"all\" onclick=\"setFilter('all')\">All</button>
                    <button class=\"tab\" data-filter=\"unassigned\" onclick=\"setFilter('unassigned')\">Unassigned</button>
                    <button class=\"tab\" data-filter=\"mine\" onclick=\"setFilter('mine')\">Assigned to me</button>
                </div>
                <div id=\"handoffs\" class=\"list\"></div>
            </div>
            <div class=\"card\">
                <div class=\"conversation-head\">
                    <div>
                        <h2 id=\"title\">Conversation</h2>
                        <div id=\"conversationMeta\" class=\"muted\"></div>
                    </div>
                    <div class=\"controls\">
                        <button onclick=\"assignCurrent()\">Assign To Me</button>
                        <button class=\"secondary\" onclick=\"resumeCurrent()\">Resume Bot</button>
                    </div>
                </div>
                <div class=\"summary-box\" id=\"summary\">Select a handoff to view its summary.</div>
                <div class=\"transcript-box\">
                    <h3>Conversation Transcript</h3>
                    <div id=\"messages\" class=\"messages\"></div>
                </div>
                <div style=\"height:12px\"></div>
                <textarea id=\"reply\" placeholder=\"Type human reply to customer\"></textarea>
                <div class=\"canned-section\">
                    <div class=\"canned-label\">Quick replies</div>
                    <div class=\"canned-btns\">
                        <button class=\"canned-btn\" onclick=\"insertCanned('Hi! I\\'m a Tailorsin team member. How can I help you today?')\">👋 Greeting</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Thank you for your patience. I\\'m looking into your order right now and will have an update for you shortly.')\">⏳ Looking into it</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Could you please share your order number so I can pull up your details?')\">🔍 Ask order number</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Your order is currently being processed and will be ready for dispatch within 2–3 business days.')\">📦 Order status</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Your measurements are safely saved in our system. Would you like to make any updates before we proceed?')\">📏 Measurements saved</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('I\\'m escalating your case to our senior tailor team and will get back to you within the hour.')\">🔺 Escalating</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Is there anything else I can help you with today?')\">✅ Anything else?</button>
                        <button class=\"canned-btn\" onclick=\"insertCanned('Thank you for reaching out to Tailorsin. Have a great day!')\">👋 Closing</button>
                    </div>
                </div>
                <div class=\"controls\" style=\"margin-top:12px\">
                    <button onclick=\"sendReply()\">Send Reply</button>
                </div>
            </div>
            <div class=\"card\">
                <h2>Audit Trail</h2>
                <div id=\"audit\" class=\"audit-list\"></div>
                <div style=\"margin-top:18px\"><h3>WhatsApp History (CRM)</h3><button class=\"secondary\" style=\"margin-bottom:10px\" onclick=\"loadWhatsappHistory()\">Load History</button><div id=\"waHistory\" class=\"audit-list\"></div></div>
            </div>
        </div>
    </div>
    <script>
        let currentId = null;
        let cachedHandoffs = [];
        let refreshTimer = null;
        let activeFilter = 'all';

        function headers() {
            return {
                'Content-Type': 'application/json',
                'x-support-token': document.getElementById('token').value,
            };
        }

        function formatTime(value) {
            if (!value) return 'n/a';
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return value;
            return date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
        }

        function flash(message, kind = 'info') {
            const el = document.getElementById('flash');
            el.textContent = message;
            el.className = kind === 'error' ? 'warn' : kind === 'ok' ? 'ok' : 'muted';
        }

        function setSelectedCard() {
            document.querySelectorAll('.item').forEach(node => {
                node.classList.toggle('active', node.dataset.customerId === currentId);
            });
        }

        function setFilter(filter) {
            activeFilter = filter;
            document.querySelectorAll('.tab').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.filter === filter);
            });
            renderHandoffList();
        }

        function filteredHandoffs() {
            const agentName = document.getElementById('agent').value.trim().toLowerCase();
            if (activeFilter === 'unassigned') return cachedHandoffs.filter(h => !h.assigned_to);
            if (activeFilter === 'mine') return cachedHandoffs.filter(h => h.assigned_to && h.assigned_to.toLowerCase() === agentName);
            return cachedHandoffs;
        }

        function insertCanned(text) {
            const el = document.getElementById('reply');
            const pos = el.selectionStart ?? el.value.length;
            const before = el.value.slice(0, pos);
            const after = el.value.slice(el.selectionEnd ?? pos);
            el.value = before + (before && !before.endsWith(' ') ? ' ' : '') + text + (after ? ' ' + after : '');
            el.focus();
            el.selectionStart = el.selectionEnd = el.value.length;
        }

        function renderMessages(messages) {
            const box = document.getElementById('messages');
            box.innerHTML = '';
            if (!messages || messages.length === 0) {
                box.innerHTML = '<div class="empty">No transcript available yet.</div>';
                return;
            }
            messages.forEach(message => {
                const div = document.createElement('div');
                div.className = `msg ${message.role || 'assistant'}`;
                div.innerHTML = `
                    <div class="msg-role">${message.role || 'unknown'}</div>
                    <div class="msg-content"></div>
                `;
                div.querySelector('.msg-content').textContent = message.content || '';
                box.appendChild(div);
            });
            box.scrollTop = box.scrollHeight;
        }

        function renderAudit(auditLog) {
            const box = document.getElementById('audit');
            box.innerHTML = '';
            if (!auditLog || auditLog.length === 0) {
                box.innerHTML = '<div class="empty">No audit events recorded yet.</div>';
                return;
            }
            [...auditLog].reverse().forEach(event => {
                const div = document.createElement('div');
                div.className = 'audit-item';
                const details = event.details ? JSON.stringify(event.details, null, 2) : '';
                div.innerHTML = `
                    <div class="audit-action">${event.action}</div>
                    <div class="audit-time">${event.actor || 'system'} · ${formatTime(event.timestamp)}</div>
                    <pre class="muted mono" style="white-space:pre-wrap; margin:10px 0 0">${details}</pre>
                `;
                box.appendChild(div);
            });
        }

        function showConversation(handoff) {
            currentId = handoff.customer_id;
            document.getElementById('title').innerText = `Conversation: ${handoff.customer_name || handoff.customer_id}`;
            document.getElementById('conversationMeta').innerText = `${handoff.segment || 'unknown'} · Assigned to ${handoff.assigned_to || 'unassigned'} · Requested ${formatTime(handoff.requested_at)}`;
            document.getElementById('summary').textContent = handoff.summary || 'No summary available.';
            renderMessages(handoff.messages || []);
            renderAudit(handoff.audit_log || []);
            setSelectedCard();
        }

        function renderHandoffList() {
            const el = document.getElementById('handoffs');
            el.innerHTML = '';
            const visible = filteredHandoffs();
            if (visible.length === 0) {
                el.innerHTML = '<div class="empty">No handoffs match this filter.</div>';
                return;
            }
            visible.forEach(handoff => {
                const div = document.createElement('div');
                div.className = 'item';
                div.dataset.customerId = handoff.customer_id;
                div.innerHTML = `
                    <div class="item-head">
                        <strong>${handoff.customer_name || 'Unknown Customer'}</strong>
                        <span class="pill">${handoff.segment || 'unknown'}</span>
                    </div>
                    <div class="muted mono">${handoff.customer_id}</div>
                    <div class="meta-row muted" style="margin-top:8px">
                        <span>Assigned: ${handoff.assigned_to || 'unassigned'}</span>
                        <span>Last active: ${formatTime(handoff.last_activity_at)}</span>
                    </div>
                `;
                div.onclick = () => showConversation(handoff);
                el.appendChild(div);
            });
            setSelectedCard();
        }

        async function loadHandoffs({ keepSelection = true } = {}) {
            try {
                const res = await fetch('/handoffs/open', { headers: headers() });
                const data = await res.json();
                if (!res.ok) {
                    flash(data.detail || 'Failed to load handoffs', 'error');
                    return;
                }
                cachedHandoffs = data.handoffs || [];
                document.getElementById('queueCount').textContent = `${cachedHandoffs.length} open handoff${cachedHandoffs.length === 1 ? '' : 's'}`;
                document.getElementById('lastRefresh').textContent = `Refreshed ${formatTime(new Date().toISOString())}`;

                renderHandoffList();

                if (cachedHandoffs.length === 0) {
                    if (!keepSelection) currentId = null;
                    if (!currentId) {
                        document.getElementById('title').innerText = 'Conversation';
                        document.getElementById('conversationMeta').innerText = '';
                        document.getElementById('summary').textContent = 'Select a handoff to view its summary.';
                        renderMessages([]);
                        renderAudit([]);
                    }
                    flash('Queue refreshed', 'ok');
                    return;
                }

                const selected = keepSelection ? cachedHandoffs.find(item => item.customer_id === currentId) : null;
                if (selected) {
                    showConversation(selected);
                } else if (!currentId && cachedHandoffs[0]) {
                    showConversation(cachedHandoffs[0]);
                } else {
                    setSelectedCard();
                }
                flash('Queue refreshed', 'ok');
            } catch (error) {
                flash(error.message || 'Unexpected error', 'error');
            }
        }

        async function assignCurrent() {
            if (!currentId) return flash('Select a handoff first', 'error');
            const agent = document.getElementById('agent').value.trim();
            if (!agent) return flash('Enter an agent name first', 'error');
            const res = await fetch(`/handoffs/${encodeURIComponent(currentId)}/assign`, {
                method: 'POST',
                headers: headers(),
                body: JSON.stringify({ agent_name: agent })
            });
            const data = await res.json();
            if (!res.ok) return flash(data.detail || 'Failed to assign case', 'error');
            flash(`Assigned to ${agent}`, 'ok');
            await loadHandoffs();
        }

        async function sendReply() {
            if (!currentId) return flash('Select a handoff first', 'error');
            const agent = document.getElementById('agent').value.trim();
            const message = document.getElementById('reply').value.trim();
            if (!agent) return flash('Enter an agent name first', 'error');
            if (!message) return flash('Reply message cannot be empty', 'error');
            const res = await fetch(`/handoffs/${encodeURIComponent(currentId)}/reply`, {
                method: 'POST',
                headers: headers(),
                body: JSON.stringify({ message, agent_name: agent })
            });
            const data = await res.json();
            if (!res.ok) return flash(data.detail || 'Failed to send reply', 'error');
            document.getElementById('reply').value = '';
            flash('Human reply sent to customer', 'ok');
            await loadHandoffs();
        }

        async function resumeCurrent() {
            if (!currentId) return flash('Select a handoff first', 'error');
            const agent = document.getElementById('agent').value.trim();
            const res = await fetch(`/handoffs/${encodeURIComponent(currentId)}/resume`, {
                method: 'POST',
                headers: headers(),
                body: JSON.stringify({ agent_name: agent || null })
            });
            const data = await res.json();
            if (!res.ok) return flash(data.detail || 'Failed to resume bot', 'error');
            currentId = null;
            flash('Conversation returned to bot', 'ok');
            await loadHandoffs({ keepSelection: false });
        }

        function scheduleAutoRefresh() {
            if (refreshTimer) clearInterval(refreshTimer);
            if (!document.getElementById('autoRefresh').checked) return;
            refreshTimer = setInterval(() => loadHandoffs(), 10000);
        }

        async function loadWhatsappHistory() {
            if (!currentId) return flash('Select a handoff first', 'error');
            const mobile = currentId.replace('whatsapp:+', '');
            const box = document.getElementById('waHistory');
            box.innerHTML = '<div class="muted">Loading...</div>';
            try {
                const res = await fetch(`/crm/whatsapp-history?mobile=${encodeURIComponent(mobile)}`, { headers: headers() });
                const data = await res.json();
                if (!res.ok) { box.innerHTML = `<div class="warn">Error: ${data.detail || 'Failed'}</div>`; return; }
                const records = data.history || [];
                if (records.length === 0) { box.innerHTML = '<div class="empty">No WhatsApp history found.</div>'; return; }
                box.innerHTML = '';
                records.forEach(record => {
                    const div = document.createElement('div');
                    div.className = 'audit-item';
                    div.innerHTML = `
                        <div class="audit-action">${record.message || record.body || ''}</div>
                        <div class="audit-time">${record.direction || ''} · ${formatTime(record.created_at || record.timestamp || '')}</div>
                    `;
                    box.appendChild(div);
                });
            } catch (err) {
                box.innerHTML = `<div class="warn">Error: ${err.message}</div>`;
            }
        }

        document.getElementById('autoRefresh').addEventListener('change', scheduleAutoRefresh);
        window.addEventListener('load', async () => {
            scheduleAutoRefresh();
            const cfg = await fetch('/support/config').then(r=>r.json()).catch(()=>({}));
            if (cfg.token) document.getElementById('token').value = cfg.token;
            await loadHandoffs();
        });
    </script>
</body>
</html>
    """

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    parsed_form = dict(form_data)
    host_header = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").lower()
    allow_invalid_for_ngrok = "ngrok" in host_header

    signature_present = bool(request.headers.get("X-Twilio-Signature", ""))
    is_twilio_request = await verify_twilio_request(request, parsed_form) if signature_present else False

    if signature_present and not is_twilio_request:
        if ALLOW_INVALID_TWILIO_SIGNATURE or allow_invalid_for_ngrok:
            logger.warning(
                "Proceeding with invalid Twilio signature. sender=%s allow_env=%s allow_ngrok=%s host=%s",
                parsed_form.get("From", ""),
                ALLOW_INVALID_TWILIO_SIGNATURE,
                allow_invalid_for_ngrok,
                host_header,
            )
        else:
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # Return JSON only when explicitly enabled for local/manual debug requests.
    # Twilio expects TwiML XML. Returning JSON to Twilio leads to "no reply" in WhatsApp.
    is_local_request = (request.client.host if request.client else "") in {"127.0.0.1", "::1", "localhost"}
    test_mode = WEBHOOK_JSON_TEST_MODE and (not signature_present) and is_local_request

    incoming_msg = parsed_form.get("Body", "").strip()
    sender_id = parsed_form.get("From", "")
    sender_name = parsed_form.get("ProfileName", "Customer")

    if not incoming_msg or not sender_id:
        if test_mode:
            return JSONResponse({"status": "no_message"})
        return twiml_message_response()

    logger.info(f"Message from {sender_name} ({sender_id}): {incoming_msg}")

    mobile = normalize_mobile(sender_id)

    conversation = await store.get_or_create_conversation(sender_id)

    session_expired = is_session_expired(conversation)
    if session_expired:
        logger.info(f"Session expired for {sender_id}; resetting conversation state.")
        reset_conversation_state(conversation)
        add_audit_event(conversation, "session_expired", "system", {"customer_id": sender_id})

    if conversation["handoff_active"]:
        conversation["messages"].append({"role": "user", "content": incoming_msg})
        conversation["last_activity_at"] = utc_now()
        add_audit_event(conversation, "customer_message_during_handoff", "customer", {"content": incoming_msg[:160]})
        await store.save_conversation(sender_id, conversation)
        logger.info(f"Handoff active for {sender_id}; acknowledging while human team handles conversation.")
        waiting_message = (
            "👤 Our human support team has your latest message and will continue with you shortly."
        )
        if test_mode:
            return JSONResponse({"status": "handoff_active", "message": waiting_message})
        return twiml_message_response(waiting_message)

    conversation["messages"].append({"role": "user", "content": incoming_msg})
    conversation["last_activity_at"] = utc_now()
    add_audit_event(conversation, "customer_message", "customer", {"content": incoming_msg[:160]})

    # --- Lazy-load customer profile if not yet set ---
    if conversation["customer_profile"] is None:
        try:
            conversation["customer_profile"] = classify_customer_profile(mobile)
        except Exception as exc:
            logger.exception("Failed to classify customer profile; defaulting to new_user. mobile=%s error=%s", mobile, exc)
            conversation["customer_profile"] = {
                "mobile": mobile,
                "segment": "new_user",
                "customer_name": sender_name,
                "reason": "fallback_after_classification_error",
                "orders_count": 0,
                "active_orders": 0,
            }

    profile = conversation["customer_profile"]

    # --- Navigation: show main menu ---
    if is_menu_request(incoming_msg) or normalize_command(incoming_msg) == "10":
        is_return_to_menu = normalize_command(incoming_msg) == "10"
        menu = build_segment_menu(profile, include_greeting=not is_return_to_menu)
        if session_expired:
            menu = (
                "⏳ Your previous session ended due to inactivity.\n\n"
                + menu
            )
        conversation["agent_context"] = {}
        conversation["messages"].append({"role": "assistant", "content": menu})
        add_audit_event(conversation, "menu_shown", "bot", {"trigger": normalize_command(incoming_msg)})
        await store.save_conversation(sender_id, conversation)
        if test_mode:
            return JSONResponse({"message": menu, "status": "menu_shown"})
        return twiml_message_response(menu)

    # --- Navigation: request human handoff ---
    if is_human_request(incoming_msg) or normalize_command(incoming_msg) == "9":
        handoff_msg = (
            "👤 Connecting you to a Tailorsin team member now.\n\n"
            "Our team will respond shortly. Thank you for your patience!"
        )
        conversation["handoff_active"] = True
        conversation["agent_context"] = {}
        conversation["messages"].append({"role": "assistant", "content": handoff_msg})
        add_audit_event(conversation, "handoff_requested", "customer", {"trigger": normalize_command(incoming_msg)})
        await notify_human_team(sender_id, sender_name, conversation, profile)
        await store.save_conversation(sender_id, conversation)
        if test_mode:
            return JSONResponse({"message": handoff_msg, "status": "handoff_initiated"})
        return twiml_message_response(handoff_msg)

    # If this is the first message, show menu and exit
    if len(conversation["messages"]) == 1:
        menu = build_segment_menu(profile)
        if session_expired:
            menu = (
                "⏳ Your previous session ended due to inactivity.\n\n"
                + menu
            )
        conversation["agent_context"] = {}
        conversation["messages"].append({"role": "assistant", "content": menu})
        add_audit_event(conversation, "greeting_sent", "bot", {"segment": profile.get("segment")})
        await store.save_conversation(sender_id, conversation)
        if test_mode:
            return JSONResponse({
                "message": menu,
                "status": "greeting_sent",
                "customer_segment": profile.get("segment"),
                "customer_name": profile.get("customer_name"),
            })
        logger.info(f"Sent greeting to {sender_name}")
        return twiml_message_response(menu)

    # Process through LangGraph agent for subsequent messages
    persisted_context = dict(conversation.get("agent_context") or {})
    persisted_context.update({
        "customer_segment": profile.get("segment", "new_user"),
        "customer_name": profile.get("customer_name"),
    })

    agent_state = await agent.workflow.ainvoke({
        "messages": conversation["messages"],
        "sentiment_score": 0,
        "needs_human": False,
        "context": persisted_context,
        "conversation_id": sender_id
    })

    post_response_handoff = bool(agent_state["context"].pop("handoff_after_response", False))
    conversation["agent_context"] = {
        key: value
        for key, value in agent_state["context"].items()
        if key not in {"customer_segment", "customer_name"}
    }

    if agent_state["needs_human"]:
        conversation["handoff_active"] = True
        add_audit_event(conversation, "handoff_requested", "bot", {"reason": "intent_or_sentiment"})
        await notify_human_team(sender_id, sender_name, conversation, profile)

    assistant_response = agent_state["messages"][-1]["content"]
    conversation["messages"].append({"role": "assistant", "content": assistant_response})
    conversation["last_activity_at"] = utc_now()
    add_audit_event(conversation, "bot_response", "bot", {"preview": assistant_response[:160]})

    if post_response_handoff and not conversation["handoff_active"]:
        conversation["handoff_active"] = True
        add_audit_event(conversation, "handoff_requested", "bot", {"reason": "post_response_flow"})
        await notify_human_team(sender_id, sender_name, conversation, profile)

    await store.save_conversation(sender_id, conversation)

    if test_mode:
        return JSONResponse({"message": assistant_response, "status": "message_sent", "needs_human": agent_state["needs_human"]})

    logger.info(f"Replying to {sender_name}: {assistant_response[:120]}")
    return twiml_message_response(assistant_response)

async def notify_human_team(sender_id: str, sender_name: str, conversation: dict, customer_profile: dict | None = None):
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    customer_profile = customer_profile or {}
    summary = await agent.summarize_handoff(
        conversation["messages"],
        customer_name=customer_profile.get("customer_name") or sender_name,
        customer_segment=customer_profile.get("segment"),
    )

    conversation["handoff_summary"] = summary
    conversation["handoff_requested_at"] = utc_now()
    add_audit_event(conversation, "handoff_summary_created", "bot", {"summary_preview": summary[:160]})

    handoff_message = (
        f"🚨 Tailorsin handoff required\n\n"
        f"Customer Name: {customer_profile.get('customer_name') or sender_name}\n"
        f"Customer WhatsApp: {sender_id}\n"
        f"Segment: {customer_profile.get('segment', 'unknown')}\n\n"
        f"{summary}\n\n"
        "Please continue the conversation with this customer using the shared support workflow."
    )

    if HUMAN_HANDOFF_WHATSAPP_NUMBER:
        send_whatsapp_message(HUMAN_HANDOFF_WHATSAPP_NUMBER, handoff_message)

    if slack_webhook:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(slack_webhook, json={"text": handoff_message})

    if not slack_webhook and not HUMAN_HANDOFF_WHATSAPP_NUMBER:
        logger.warning("No human handoff destination configured. Set SLACK_WEBHOOK_URL or HUMAN_HANDOFF_WHATSAPP_NUMBER.")

@app.get("/conversations/{customer_id}")
async def get_conversation(customer_id: str):
    conversation = await store.get_conversation(customer_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/support/config")
async def support_config():
    configured = SUPPORT_API_TOKEN and SUPPORT_API_TOKEN != SUPPORT_TOKEN_PLACEHOLDER
    return {"token": SUPPORT_API_TOKEN if configured else ""}


@app.get("/crm/whatsapp-history")
async def crm_whatsapp_history(mobile: str, x_support_token: str | None = Header(default=None)):
    ensure_support_access(x_support_token)
    import httpx as _httpx
    clean_mobile = mobile.replace("whatsapp:+", "").replace("+", "").strip()
    crm_url = f"https://crm.tailorsin.com/tailorsin-api/api/clientwhatsapp.php?mobile={clean_mobile}"
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(crm_url)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}
        history = data if isinstance(data, list) else data.get("data") or data.get("history") or data.get("messages") or []
        return {"mobile": clean_mobile, "history": history}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CRM error: {exc}")


@app.get("/handoffs/open")
async def list_open_handoffs(x_support_token: str | None = Header(default=None)):
    ensure_support_access(x_support_token)
    open_handoffs = []
    for item in await store.list_open_handoffs():
        customer_id = item["customer_id"]
        conversation = item["conversation"]
        profile = conversation.get("customer_profile") or {}
        open_handoffs.append({
            "customer_id": customer_id,
            "customer_name": profile.get("customer_name"),
            "segment": profile.get("segment"),
            "assigned_to": conversation.get("handoff_assigned_to"),
            "requested_at": conversation.get("handoff_requested_at"),
            "last_activity_at": conversation.get("last_activity_at"),
            "summary": conversation.get("handoff_summary"),
            "messages": conversation.get("messages", [])[-20:],
            "audit_log": conversation.get("audit_log", [])[-20:],
        })
    return {"handoffs": open_handoffs}


@app.post("/handoffs/{customer_id}/assign")
async def assign_handoff(customer_id: str, payload: HandoffAssignRequest, x_support_token: str | None = Header(default=None)):
    ensure_support_access(x_support_token)
    conversation = await store.get_conversation(customer_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conversation.get("handoff_active"):
        raise HTTPException(status_code=400, detail="No active handoff for this conversation")

    conversation["handoff_assigned_to"] = payload.agent_name
    add_audit_event(conversation, "handoff_assigned", payload.agent_name, {})
    await store.save_conversation(customer_id, conversation)
    return {
        "status": "assigned",
        "customer_id": customer_id,
        "assigned_to": payload.agent_name,
    }


@app.post("/handoffs/{customer_id}/reply")
async def reply_to_customer(customer_id: str, payload: HumanReplyRequest, x_support_token: str | None = Header(default=None)):
    ensure_support_access(x_support_token)
    conversation = await store.get_conversation(customer_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conversation.get("handoff_active"):
        raise HTTPException(status_code=400, detail="Bot currently owns this conversation")

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Reply message cannot be empty")

    outbound_text = message
    if payload.agent_name:
        conversation["handoff_assigned_to"] = payload.agent_name

    sent = send_whatsapp_message(customer_id, outbound_text)
    if not sent:
        raise HTTPException(status_code=500, detail="Failed to send WhatsApp message to customer")

    conversation["messages"].append({
        "role": "human",
        "content": outbound_text,
    })
    conversation["handoff_last_human_message_at"] = utc_now()
    conversation["last_activity_at"] = utc_now()
    add_audit_event(conversation, "human_reply_sent", payload.agent_name or "support", {"preview": outbound_text[:160]})
    await store.save_conversation(customer_id, conversation)

    return {
        "status": "sent",
        "customer_id": customer_id,
        "assigned_to": conversation.get("handoff_assigned_to"),
    }


@app.post("/handoffs/{customer_id}/resume")
async def resume_bot(customer_id: str, payload: ResumeBotRequest, x_support_token: str | None = Header(default=None)):
    ensure_support_access(x_support_token)
    conversation = await store.get_conversation(customer_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if payload.agent_name:
        conversation["handoff_assigned_to"] = payload.agent_name

    if payload.note:
        conversation["messages"].append({
            "role": "human",
            "content": f"Support note: {payload.note.strip()}",
        })

    conversation["handoff_active"] = False
    add_audit_event(conversation, "bot_resumed", payload.agent_name or "support", {"note": payload.note or ""})
    await store.save_conversation(customer_id, conversation)

    return {
        "status": "bot_resumed",
        "customer_id": customer_id,
        "assigned_to": conversation.get("handoff_assigned_to"),
    }

@app.post("/conversations/{customer_id}/handoff/reset")
async def reset_handoff(customer_id: str):
    conversation = await store.get_conversation(customer_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation["handoff_active"] = False
    add_audit_event(conversation, "handoff_reset", "system", {})
    await store.save_conversation(customer_id, conversation)
    return {"status": "handoff_reset", "customer_id": customer_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
