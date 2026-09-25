"""
LLM Client for Financial Decision Agent
Supports Groq API (gsk_...) and Google Gemini API for financial analysis, OCR image processing, and evaluation.
"""

import os
import json
import logging
import sys
import sysconfig
import importlib.util
from typing import Optional, Dict, Any
from datetime import datetime
import time
import base64
from pathlib import Path
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Prevent local directory 'code/' from shadowing Python standard library 'code' module needed by IPython/pdb
if 'code' not in sys.modules or not hasattr(sys.modules['code'], 'InteractiveConsole'):
    try:
        stdlib_path = sysconfig.get_path('stdlib')
        code_py = Path(stdlib_path) / 'code.py'
        if code_py.exists():
            spec = importlib.util.spec_from_file_location('stdlib_code', str(code_py))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['code'] = mod
    except Exception:
        pass

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Unified LLM Client supporting Groq API and Google Gemini API.
    Handles text analysis, vision OCR for financial images, and decision explanations.
    """
    
    def __init__(self, transcript_path: str = "code/logs/chat_transcript.txt",
                 model_name: Optional[str] = None):
        """
        Initialize LLM client using GROQ_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY.
        """
        self.api_key = os.getenv('GROQ_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        self.is_groq = False
        if self.api_key and (self.api_key.startswith('gsk_') or os.getenv('GROQ_API_KEY')):
            self.is_groq = True
            self.provider = "Groq API"
            self.model_name = os.getenv('GROQ_MODEL') or model_name or "groq/compound"
            self.model = None
            logger.info(f"[OK] Groq API client initialized with model: {self.model_name}")
        elif self.api_key and HAS_GEMINI:
            self.provider = "Google Gemini API"
            self.model_name = os.getenv('GEMINI_MODEL') or model_name or "gemini-2.0-flash"
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"[OK] Gemini API client initialized with model: {self.model_name}")
        else:
            self.provider = "Fallback Mode"
            self.model_name = "none"
            self.model = None
            logger.warning("No valid API key set. Running in rule-based fallback mode.")

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        
        self.transcript_path = Path(transcript_path)
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize transcript log file
        with open(self.transcript_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"LLM Session started ({self.provider}): {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"{'='*80}\n")
    
    def _log_interaction(self, role: str, content: str, tokens: Dict = None):
        """Log interaction to transcript"""
        try:
            with open(self.transcript_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[{datetime.now().isoformat()}] {role.upper()}:\n")
                display_content = content[:500] + "..." if len(content) > 500 else content
                f.write(f"{display_content}\n")
                if tokens:
                    f.write(f"[Tokens] Input: {tokens.get('input', 'N/A')}, Output: {tokens.get('output', 'N/A')}\n")
        except Exception as e:
            logger.warning(f"Failed to log interaction to transcript: {e}")

    def analyze_financial_state(self, analysis_prompt: str) -> Dict[str, Any]:
        """Ask LLM (Groq or Gemini) to analyze financial state and make recommendations."""
        self._log_interaction("user", analysis_prompt)
        
        if not self.api_key:
            return {"error": "API key not configured"}

        system_instruction = """You are an expert financial advisor analyzing whether a customer can safely afford a purchase.
Respond ONLY with valid JSON with keys:
{
  "amount_safe_to_pay": <float>,
  "affordability_status": "<affordable_now|affordable_with_plan|affordable_later|not_affordable>",
  "recommended_payment_method": "<full_payment|partial_payment|installments|wait|not_recommended>",
  "payment_plan": "<string>",
  "earliest_date_for_full_payment": "<YYYY-MM-DD or empty>",
  "spending_changes_needed": "<string>",
  "decision_explanation": "<concise explanation>"
}"""
            
        if self.is_groq:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Financial Request & Data:\n{analysis_prompt}"}
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"}
            }

            for attempt in range(3):
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=30)
                    if res.status_code == 200:
                        res_data = res.json()
                        response_text = res_data['choices'][0]['message']['content'].strip()
                        self._log_interaction("assistant", response_text)
                        
                        usage = res_data.get('usage', {})
                        in_tok = usage.get('prompt_tokens', int(len(analysis_prompt.split()) * 1.3))
                        out_tok = usage.get('completion_tokens', int(len(response_text.split()) * 1.3))
                        self.total_input_tokens += in_tok
                        self.total_output_tokens += out_tok
                        self.call_count += 1
                        
                        return json.loads(response_text)
                    elif res.status_code == 429:
                        logger.warning("Groq Rate limited (429), proceeding with rule engine decision.")
                        break
                    else:
                        logger.error(f"Groq API error {res.status_code}: {res.text}")
                        break
                except Exception as e:
                    logger.error(f"Groq request exception on attempt {attempt+1}: {e}")
                    time.sleep(2)
            return {"error": "Groq API request failed"}

        elif self.model:
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000,
                )
                full_prompt = f"{system_instruction}\n\nFinancial Request & Data:\n{analysis_prompt}"

                for attempt in range(3):
                    try:
                        response = self.model.generate_content(
                            full_prompt,
                            generation_config=generation_config
                        )
                        
                        response_text = response.text.strip()
                        if response_text.startswith('```json'):
                            response_text = response_text[7:]
                        if response_text.startswith('```'):
                            response_text = response_text[3:]
                        if response_text.endswith('```'):
                            response_text = response_text[:-3]
                        response_text = response_text.strip()
                        
                        self._log_interaction("assistant", response_text)
                        
                        self.total_input_tokens += int(len(full_prompt.split()) * 1.3)
                        self.total_output_tokens += int(len(response_text.split()) * 1.3)
                        self.call_count += 1
                        
                        return json.loads(response_text)
                    except Exception as e:
                        if '429' in str(e) or 'Quota' in str(e):
                            logger.warning(f"Rate limited (429) on attempt {attempt+1}, backing off for 12 seconds...")
                            time.sleep(12)
                        else:
                            raise e

                return {"error": "Rate limit retries exhausted"}
            except Exception as e:
                logger.error(f"Gemini API analysis failed: {e}")
                return {"error": str(e)}

        return {"error": "No LLM engine available"}

    def extract_amount_from_image(self, image_path: str) -> Optional[float]:
        """Extract monetary amount from image using Vision API."""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                image_bytes = f.read()
            
            suffix = path.suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(suffix, 'image/jpeg')
            
            self._log_interaction("user", f"Extract amount from image: {path.name}")
            prompt_text = "Extract the main monetary amount shown in this document. Return ONLY the numeric value (e.g. 1500, 2500.50). If no amount is visible, return 'NOT_FOUND'."
            
            if self.is_groq:
                b64_str = base64.b64encode(image_bytes).decode('utf-8')
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "groq/compound",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_str}"}}
                            ]
                        }
                    ]
                }
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=20)
                    if res.status_code == 200:
                        res_data = res.json()
                        response_text = res_data['choices'][0]['message']['content'].strip()
                        self._log_interaction("assistant", response_text)
                        
                        usage = res_data.get('usage', {})
                        self.total_input_tokens += usage.get('prompt_tokens', 200)
                        self.total_output_tokens += usage.get('completion_tokens', 10)
                        self.call_count += 1
                        
                        if 'NOT_FOUND' in response_text:
                            return None
                        amount_str = response_text.replace('$', '').replace(',', '').strip()
                        return float(amount_str)
                except Exception as e:
                    logger.warning(f"Groq Vision OCR failed: {e}")
                    return None

            elif self.model:
                for attempt in range(3):
                    try:
                        response = self.model.generate_content([
                            prompt_text,
                            {"mime_type": media_type, "data": image_bytes}
                        ])
                        response_text = response.text.strip()
                        self._log_interaction("assistant", response_text)
                        
                        self.total_input_tokens += 200
                        self.total_output_tokens += len(response_text.split())
                        self.call_count += 1
                        
                        if 'NOT_FOUND' in response_text:
                            return None
                            
                        amount_str = response_text.replace('$', '').replace(',', '').strip()
                        return float(amount_str)
                    except Exception as e:
                        if '429' in str(e) or 'Quota' in str(e):
                            logger.warning(f"Rate limited (429) on attempt {attempt+1}, backing off for 12 seconds...")
                            time.sleep(12)
                        else:
                            raise e
            return None
            
        except Exception as e:
            logger.error(f"Vision API error processing {image_path}: {e}")
            return None

    def get_token_usage_report(self) -> Dict[str, Any]:
        """Get token usage summary report."""
        avg_input = self.total_input_tokens / max(1, self.call_count)
        avg_output = self.total_output_tokens / max(1, self.call_count)
        
        cost_per_in = 0.0000001 if self.is_groq else 0.000000075
        cost_per_out = 0.0000002 if self.is_groq else 0.0000003
        
        return {
            'provider': self.provider,
            'model': self.model_name,
            'total_calls': self.call_count,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_input_tokens_per_request': round(avg_input, 2),
            'avg_output_tokens_per_request': round(avg_output, 2),
            'estimated_total_cost_usd': round((self.total_input_tokens * cost_per_in) + (self.total_output_tokens * cost_per_out), 4)
        }
