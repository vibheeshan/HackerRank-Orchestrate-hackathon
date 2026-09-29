# Buy or Wait? — AI-Powered Financial Decision Agent

[![Challenge](https://img.shields.io/badge/HackerRank-Orchestrate%20Sept%202026-blue)](https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait)
[![Accuracy](https://img.shields.io/badge/Evaluation%20Accuracy-100%25-brightgreen)](#-evaluation--benchmark-performance)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Google%20Gemini-orange)](https://groq.com/)

> **HackerRank Orchestrate Challenge (September 2026)**  
> An autonomous AI agent designed to evaluate personal purchase and payment requests, reconstructing cash flows, enforcing financial safety bounds, and recommending personalized payment plans.

---

## 🏆 Evaluation & Benchmark Performance

Our hybrid decision architecture achieves **perfect accuracy** across the evaluation benchmark:

| Metric | Result | Target | Status |
|---|---|---|---|
| **Overall Accuracy** | **100.00%** | > 95% | ✅ Passed |
| **Exact Matches** | **25 / 25** | 25 / 25 | ✅ Passed |
| **Amount Errors** | **$0.00** | $0.00 | ✅ Passed |
| **Status Mismatches** | **0** | 0 | ✅ Passed |
| **Method Mismatches** | **0** | 0 | ✅ Passed |
| **Total Mismatches** | **0** | 0 | ✅ Passed |
| **Evaluated Requests** | **250 / 250** | 250 | ✅ Fully Generated |

---

## 🎯 Problem Statement

When a user asks: **"Can I afford this laptop?"**, an intelligent response requires looking far beyond the user's current account balance.

The **Buy or Wait?** agent reconstructs the user's complete financial picture by integrating:
- **Historical & Scheduled Cash Flows**: Salary dates, rent, subscriptions, utility bills, debt repayments.
- **Pending Debits & Credits**: Reserving pending debits conservatively while excluding unconfirmed credits or speculative gains.
- **Foreign Exchange Rates**: Fixed dated conversion rates for foreign-currency events (USD, EUR, INR, ZAR, IDR).
- **User Priorities & Minimum Balance**: Strict enforcement of `minimum_balance_to_keep` and user risk thresholds.
- **Merchant Payment Options**: Structured installment plans, down payments, partial payment schedules.
- **Multimodal & Unstructured Evidence**: Processing receipt images, billing statements, and chat messages.

For every request, the agent outputs one of four primary affordability statuses:
1. **`affordable_now`**: Full payment is safe on the request date without any spending changes.
2. **`affordable_with_plan`**: Full payment is safe via partial payments, merchant installments, or approved flexible spending adjustments.
3. **`affordable_later`**: Full payment will become safe on a projected future date within the forecast period.
4. **`not_affordable`**: Full payment cannot be safely completed within the forecast period.

---

## 🏗 System Architecture

The solution uses a **two-stage hybrid architecture**:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA & CONTEXT INGESTION                           │
│  - Financial Profiles (Minimum Balance, Preferences, Max Installments)          │
│  - Financial Events & Recurrence Forecasts                                      │
│  - Merchant Payment Options & Exchange Rates                                    │
│  - Multimodal Evidence (Image receipts & contextual text messages)              │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       2. DETERMINISTIC FINANCIAL CALCULATOR                     │
│  - Daily Cash Flow Projection & Essential Expense Reservation                   │
│  - Safe Amount Calculation (0 <= amount_safe_to_pay <= requested_amount)        │
│  - Installment Option Validation & Preference Matching                          │
│  - Flexible Category Spending Adjustment Optimizer (stop / reduce_to)           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       3. LLM INTELLIGENT REFINEMENT ENGINE                      │
│  - Primary: Groq API (llama-3.3-70b-versatile)                                  │
│  - Fallback: Google Gemini API (gemini-2.5-flash)                               │
│  - Generates natural, grounded decision explanations                             │
│  - Validates complex edge-case conflict resolutions                             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             4. OUTPUT GENERATION                                │
│  - dataset/output.csv (250 evaluated predictions)                               │
│  - code/evaluation/usage_report.md (Token & cost breakdown)                     │
│  - log.txt (Full audit transcript)                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26
pip install -r requirements.txt
```

### 2. Configure API Credentials

Set your API key for Groq (Primary) or Google Gemini (Fallback):

**Linux / macOS:**
```bash
export GROQ_API_KEY="your-groq-api-key"
# or
export GOOGLE_API_KEY="your-google-gemini-key"
```

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your-groq-api-key"
# or
$env:GOOGLE_API_KEY="your-google-gemini-key"
```

**Windows CMD:**
```cmd
set GROQ_API_KEY=your-groq-api-key
```

Alternatively, copy `.envexample` to `.env` and fill in your credentials:
```bash
cp .envexample .env
```

### 3. Run Pipeline & Evaluate

Generate final predictions for all 250 requests:
```bash
python main.py
```

Run the validation harness to score predictions against sample ground truth:
```bash
python evaluation_harness.py
```

---

## 📊 Dataset & Output Schema

The agent processes `dataset/requests.csv` and outputs `output.csv` with the exact schema:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

### Column Specifications

| Column Name | Type | Description / Format |
|---|---|---|
| `request_id` | `string` | Unique identifier of the evaluated request |
| `amount_safe_to_pay` | `float` | Maximum amount safe to pay today (`0 <= safe <= requested`) |
| `affordability_status` | `enum` | `affordable_now` \| `affordable_with_plan` \| `affordable_later` \| `not_affordable` |
| `recommended_payment_method` | `enum` | `full_payment` \| `partial_payment` \| `installments` \| `wait` \| `not_recommended` |
| `payment_plan` | `string` | Pipe-separated `<YYYY-MM-DD>:<amount>` payments, or `none` |
| `earliest_date_for_full_payment` | `date` | Conservative date full payment becomes safe (`YYYY-MM-DD` or empty) |
| `spending_changes_needed` | `string` | Up to 3 `stop:<event_id>` or `reduce_to:<event_id>:<amount>` actions, or `none` |
| `decision_explanation` | `string` | Concise, grounded financial explanation supporting the decision |

---

## 📂 Repository File Layout

```text
.
├── AGENTS.md                         # Protocol rules & transcript specifications for AI agents
├── CLAUDE.md                         # Rule imports for harness compatibility
├── log.txt                           # Master conversation & execution audit transcript
├── problem_statement.md              # Official challenge statement & financial decision rules
├── README.md                         # Project documentation & execution guide
├── PROJECT_PRESENTATION.txt         # Full 4,300+ word project breakdown & architectural reference
├── requirements.txt                  # Python dependencies
├── main.py                           # Root terminal entry point (`python main.py`)
├── evaluation_harness.py             # Evaluation & verification script
├── output.csv                        # Final generated predictions (250 rows)
├── sample_output.csv                 # Reference solved predictions
├── code.zip                          # Official HackerRank submission ZIP package
├── code/                             # Core implementation package
│   ├── __init__.py
│   ├── main.py                       # Decision engine, financial calculator & LLM pipeline
│   ├── gemini_client.py              # LLM client abstraction (Groq + Gemini integration)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── main.py                   # Evaluation execution entry
│   │   ├── evaluation_report.txt     # Benchmark accuracy report
│   │   └── usage_report.md           # Mandatory token usage & cost summary
│   └── logs/
│       ├── chat_transcript.txt       # LLM chat transcript
│       └── execution.log             # Pipeline execution logs
└── dataset/                          # Input data directory
    ├── requests.csv                  # 250 evaluation requests
    ├── output.csv                    # Template output CSV
    ├── sample_requests.csv           # 25 sample requests with ground truth
    ├── financial_profiles.csv        # User balances, preferences, priorities
    ├── financial_events.csv          # Historical & pending financial transactions
    ├── request_payment_options.csv   # Merchant payment options per request
    ├── exchange_rates.csv            # Dated currency conversion rates
    ├── messages.csv                  # User context messages
    ├── images.csv                    # Image metadata & OCR links
    └── media/images/                 # Image assets (image_01.png - image_16.png)
```

---

## 💰 Token Usage & Cost Analysis

As specified in challenge rule §6.5, model resource usage is tracked in `code/evaluation/usage_report.md`:

- **Model Provider**: Groq API / Google Gemini Fallback
- **Primary Model**: `llama-3.3-70b-versatile`
- **Total Requests Evaluated**: 250
- **Total Model API Calls**: 250
- **Average Tokens / Request**: ~875 tokens
- **Estimated Total Cost**: **~$0.03 USD**

---

## 📤 HackerRank Submission Guidelines

Upload your submission directly to the official platform:
👉 **[HackerRank Submission Page](https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission)**

Upload the following three files:
1. **Code Zip (`code.zip`)**: Package containing `code/`, `main.py`, `evaluation_harness.py`, `README.md`, `requirements.txt`, and `code/evaluation/usage_report.md`.
2. **Predictions CSV (`output.csv`)**: Populated predictions file for all 250 evaluation requests.
3. **Chat Transcript (`log.txt` / `chat_transcript.txt`)**: Complete development log file.
