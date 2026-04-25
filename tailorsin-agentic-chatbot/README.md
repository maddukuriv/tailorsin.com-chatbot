# Tailorsin WhatsApp Agent

FastAPI backend for a WhatsApp-based e-tailoring chatbot using Twilio WhatsApp API.

## Setup

1. Navigate to `tailorsin-agentic-chatbot`:
   ```bash
   cd tailorsin-agentic-chatbot
   ```

2. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Add credentials to `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   REDIS_URL=redis://localhost:6379/0
   SUPPORT_API_TOKEN=replace-with-a-secure-random-token
   HUMAN_HANDOFF_WHATSAPP_NUMBER=whatsapp:+91XXXXXXXXXX
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
   ```

4. Start Redis locally:
   ```bash
   redis-server
   ```

## Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Twilio WhatsApp Setup

1. **Get Twilio account** at [twilio.com](https://www.twilio.com/)
2. **Enable WhatsApp** in your Twilio Console:
   - Go to **Messaging** → **Settings** → **WhatsApp**
   - Click **Get Started** with WhatsApp
3. **Get your credentials**:
   - **Account SID**: From your dashboard
   - **Auth Token**: From your dashboard
   - **WhatsApp Number**: Use the sandbox number or get approved for production
4. **Set webhook URL** in Twilio:
   - Go to **WhatsApp** settings
   - Set **WHEN A MESSAGE COMES IN** to: `https://your-domain/webhook`
   - For local testing, use ngrok:
     ```bash
   ## Setup
     ```
   1. Navigate to the project:
## Website Frontend
      cd /Users/maddukuri/Documents/GitHub/tailorsin.com/tailorsin-agentic-chatbot
Add this WhatsApp link to your website:

   2. Install dependencies into the project virtual environment:
<a href="https://wa.me/919XXXXXXXXXX?text=Hi%20Tailorsin" target="_blank">
      /Users/maddukuri/Documents/GitHub/tailorsin.com/.venv/bin/pip install -r requirements.txt
</a>
```
   3. Add credentials to `.env`:
Replace `919XXXXXXXXXX` with your WhatsApp number.
      LLM_PROVIDER=groq
      GROQ_API_KEY=your_groq_api_key
      GROQ_MODEL=llama-3.3-70b-versatile
      OPENAI_API_KEY=your_openai_api_key
      OPENAI_MODEL=gpt-4o
## API Endpoints

      TWILIO_WHATSAPP_NUMBER=whatsapp:+19892682752
- `GET /conversations/{customer_id}` - Get conversation history
- `POST /conversations/{customer_id}/handoff/reset` - Reset handoff status
- `GET /handoffs/open` - List open support handoffs
- `POST /handoffs/{customer_id}/assign` - Assign a support agent
- `POST /handoffs/{customer_id}/reply` - Send a human reply to the customer
- `POST /handoffs/{customer_id}/resume` - Return conversation control to the bot
   4. Activate the virtual environment if you want a simpler shell session:
      ```bash
      source /Users/maddukuri/Documents/GitHub/tailorsin.com/.venv/bin/activate
      ```

   ## Run the App Locally

   You should use 3 terminals: one for Redis, one for FastAPI, and one for ngrok.

   ### Terminal 1: Start Redis

   If Redis is installed via Homebrew:

   ```bash
   brew services start redis
   ```

   Or run Redis directly:

   ```bash

## Features

   ### Terminal 2: Start FastAPI
- Intent classification (orders, pricing, fabrics, measurements)
- Redis-backed conversation persistence
   cd /Users/maddukuri/Documents/GitHub/tailorsin.com/tailorsin-agentic-chatbot
   /Users/maddukuri/Documents/GitHub/tailorsin.com/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
- Shared support workflow with assignment, human reply, and bot resume

   Do not use `--reload` unless needed. In this project it has been causing the process to exit with code `137` on the current machine.

   If port `8000` is already busy, clear it first:

   ```bash
   lsof -ti tcp:8000 | xargs kill -9
   ```

   ### Terminal 3: Start ngrok

   ```bash
   ngrok http 8000
   ```

   ngrok will give you a public HTTPS URL such as:

   ```text
   https://your-subdomain.ngrok-free.app
   ```

   ## Twilio WhatsApp Setup

   1. Create or open your Twilio account.
   2. Open the WhatsApp Sandbox or your approved WhatsApp sender configuration.
   3. Set **When a message comes in** to:
      ```text
      https://your-subdomain.ngrok-free.app/webhook
      ```
   4. Set the request method to `POST`.
- Slack and optional WhatsApp notifications for handoffs
   ## Support Dashboard

   The support dashboard is part of the same FastAPI app. There is no separate server to run.

   Open it locally:

   ```bash
   open http://127.0.0.1:8000/support
   ```

   Or in the browser using the ngrok URL:

   ```text
   https://your-subdomain.ngrok-free.app/support
   ```

   If `SUPPORT_API_TOKEN` is still set to `replace-with-a-secure-random-token`, the app treats that as a placeholder and does not enforce dashboard token validation.

   ## Quick Health Checks

   Check that the FastAPI app is up:

   ```bash
   curl http://127.0.0.1:8000/
   ```

   Expected response:

   ```json
   {"status":"healthy","service":"E-Tailoring WhatsApp Agent"}
   ```

   Verify the app can initialize with the current environment:

   ```bash
   cd /Users/maddukuri/Documents/GitHub/tailorsin.com/tailorsin-agentic-chatbot
   /Users/maddukuri/Documents/GitHub/tailorsin.com/.venv/bin/python -c "import main; print('OK')"
   ```

   ## Complete Command Summary

   Use these commands in order:

   ```bash
   # Terminal 1
   brew services start redis

   # Terminal 2
   cd /Users/maddukuri/Documents/GitHub/tailorsin.com/tailorsin-agentic-chatbot
   /Users/maddukuri/Documents/GitHub/tailorsin.com/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

   # Terminal 3
   ngrok http 8000
   ```
