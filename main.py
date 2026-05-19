import os
import sys
import json
import asyncio
import aiohttp
import aiofiles
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from itertools import cycle
import time
from datetime import datetime, date

# بارگذاری متغیرهای محیطی
load_dotenv()

# ---------------------------------------------------------
# تنظیمات و ثابت‌ها
# ---------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-31b-it")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "12"))

# محدودیت‌های API
RATE_LIMIT_PER_MINUTE = 15   # درخواست در دقیقه برای هر کلید
DAILY_QUOTA_PER_KEY = 1500   # درخواست در روز برای هر کلید

# کش برای جملات تکراری (کش دوطرفه)
STATIC_TRANSLATIONS = {
    "You are a helpful AI assistant.": "تو یک دستیار هوش مصنوعی هستی."
}

# کش پویا برای جلوگیری از ارسال تکراری متن‌های مشابه در طول اجرا
_runtime_cache: dict[str, str] = {}

# ---------------------------------------------------------
# مدیریت کلیدها با Rate Limiting دقیقه‌ای و روزانه
# ---------------------------------------------------------
class KeyManager:
    def __init__(self, api_keys: list[str]):
        self.keys = api_keys
        self.key_cycle = cycle(api_keys)
        self.lock = asyncio.Lock()
        
        # ردیابی درخواست‌ها: key -> لیست تایم‌استمپ‌های دقیقه جاری
        self.minute_requests: dict[str, list[float]] = defaultdict(list)
        
        # ردیابی درخواست‌های روزانه: key -> تعداد درخواست‌های امروز
        self.daily_counts: dict[str, int] = defaultdict(int)
        self.last_reset_date: str = date.today().isoformat()

    def _reset_daily_if_needed(self):
        """ریست شمارنده روزانه اگر روز جدید شروع شده"""
        today = date.today().isoformat()
        if self.last_reset_date != today:
            self.daily_counts.clear()
            self.last_reset_date = today

    async def get_available_key(self) -> str | None:
        """
        دریافت کلیدی که هم محدودیت دقیقه‌ای و هم روزانه را نقض نکرده باشد.
        اگر هیچ کلیدی در دسترس نبود، None برمی‌گرداند.
        """
        async with self.lock:
            self._reset_daily_if_needed()
            now = time.time()
            
            # تلاش حداکثر به اندازه تعداد کلیدها
            for _ in range(len(self.keys)):
                key = next(self.key_cycle)
                
                # پاک‌سازی تایم‌استمپ‌های قدیمی‌تر از ۶۰ ثانیه
                self.minute_requests[key] = [
                    t for t in self.minute_requests[key] if now - t < 60
                ]
                
                # بررسی محدودیت‌ها
                minute_ok = len(self.minute_requests[key]) < RATE_LIMIT_PER_MINUTE
                daily_ok = self.daily_counts[key] < DAILY_QUOTA_PER_KEY
                
                if minute_ok and daily_ok:
                    # رزرو اسلات برای این کلید
                    self.minute_requests[key].append(now)
                    self.daily_counts[key] += 1
                    return key
            
            # هیچ کلیدی در دسترس نبود
            return None

    async def wait_for_key(self, timeout: float = 300) -> str:
        """صبر کردن تا زمانی که یک کلید در دسترس قرار گیرد"""
        start = time.time()
        while time.time() - start < timeout:
            key = await self.get_available_key()
            if key:
                return key
            await asyncio.sleep(1)
        
        raise TimeoutError("⏰ تمام کلیدها به محدودیت رسیدند. لطفاً بعداً تلاش کنید.")

# ---------------------------------------------------------
# تابع ترجمه با Gemini API (مدل Gemma)
# ---------------------------------------------------------
async def translate_with_gemma(
    session: aiohttp.ClientSession,
    text: str,
    api_key: str
) -> str:
    """ارسال درخواست ترجمه به مدل Gemma via Google AI Studio"""
    
    if not text or not text.strip():
        return text

    # بررسی کش رانتایم
    if text in _runtime_cache:
        return _runtime_cache[text]

    # بررسی کش استاتیک
    if text in STATIC_TRANSLATIONS:
        return STATIC_TRANSLATIONS[text]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    
    # پرامپت مهندسی‌شده برای ترجمه دقیق
    prompt = (
        "You are a professional English-to-Persian translator. "
        "Translate the following text to natural, fluent Persian (Farsi). "
        "Rules:\n"
        "- Keep numbers, code, URLs, and mathematical symbols unchanged.\n"
        "- Preserve the original tone and meaning.\n"
        "- Output ONLY the translated text, no explanations or quotes.\n\n"
        f"Text to translate:\n{text}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 4096
        }
    }

    try:
        async with session.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=45)
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                translated = data["candidates"][0]["content"]["parts"][0]["text"]
                result = translated.strip()
                
                # ذخیره در کش رانتایم
                _runtime_cache[text] = result
                return result
            
            elif response.status == 429:
                print(f"\n⚠️ Rate limit برای کلید {api_key[:10]}...")
                return None  # Signal for retry
            
            elif response.status >= 500:
                print(f"\n⚠️ خطای سرور: {response.status}")
                return None
            
            else:
                error = await response.text()
                print(f"\n❌ خطای API {response.status}: {error[:150]}")
                return text  # Fallback به متن اصلی

    except asyncio.TimeoutError:
        print("\n⏱️ Timeout در درخواست API")
        return None
    except Exception as e:
        print(f"\n❌ خطا: {type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------
# پردازش یک خط از دیتاست JSONL
# ---------------------------------------------------------
async def process_line(
    line: str,
    session: aiohttp.ClientSession,
    key_manager: KeyManager
) -> dict | None:
    
    try:
        data = json.loads(line.strip())
        messages = data.get("messages", [])
        
        # جمع‌آوری کارهای ترجمه برای این خط
        translation_tasks = []
        
        for msg in messages:
            if "content" not in msg or not isinstance(msg["content"], str):
                continue
            
            original_text = msg["content"]
            
            # کش استاتیک: بدون ارسال به API
            if original_text in STATIC_TRANSLATIONS:
                msg["content"] = STATIC_TRANSLATIONS[original_text]
                continue
            
            # کش رانتایم: بدون ارسال به API
            if original_text in _runtime_cache:
                msg["content"] = _runtime_cache[original_text]
                continue
            
            # نیاز به ترجمه واقعی
            translation_tasks.append((msg, original_text))
        
        # اگر چیزی برای ترجمه نیست، برگردان
        if not translation_tasks:
            return data
        
        # ترجمه همزمان تمام متن‌های این خط
        for msg, text in translation_tasks:
            # دریافت کلید در دسترس (با رعایت Rate Limit)
            api_key = await key_manager.wait_for_key()
            
            # ارسال درخواست ترجمه
            result = await translate_with_gemma(session, text, api_key)
            
            # اگر ترجمه شکست خورد، متن اصلی را نگه دار
            msg["content"] = result if result else text
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"⚠️ خطای JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ خطا در پردازش: {type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------
# تابع اصلی
# ---------------------------------------------------------
async def main(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"❌ فایل ورودی یافت نشد: {input_path}")

    # استخراج کلیدها از .env
    api_keys = []
    for i in range(1, 21):  # پشتیبانی تا ۲۰ کلید
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())
    
    if not api_keys:
        raise ValueError("❌ هیچ کلید API معتبری در فایل .env یافت نشد.")
    
    print(f"🔑 {len(api_keys)} کلید API بارگذاری شد.")
    print(f"🤖 مدل: {MODEL_NAME}")
    print(f"⚡ همزمانی: {MAX_CONCURRENT}")
    print(f"📊 محدودیت: {RATE_LIMIT_PER_MINUTE}/دقیقه، {DAILY_QUOTA_PER_KEY}/روز برای هر کلید")

    key_manager = KeyManager(api_keys)
    stats = {"processed": 0, "errors": 0, "cached": 0, "api_calls": 0}

    async with aiohttp.ClientSession() as session:
        # باز کردن فایل خروجی
        with open(output_file, "w", encoding="utf-8") as fout:
            
            # شمارش خطوط برای نوار پیشرفت
            total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
            
            with open(input_file, "r", encoding="utf-8") as fin:
                pbar = tqdm(desc="🔄 ترجمه", total=total_lines, unit="line")
                
                for line in fin:
                    if not line.strip():
                        pbar.update(1)
                        continue
                    
                    # شمارش متن‌های کش‌شده قبل از پردازش (برای آمار)
                    data_pre = json.loads(line.strip()) if line.strip() else {}
                    cached_count = sum(
                        1 for msg in data_pre.get("messages", [])
                        if msg.get("content") in STATIC_TRANSLATIONS or msg.get("content") in _runtime_cache
                    )
                    
                    result = await process_line(line, session, key_manager)
                    
                    if result:
                        # نوشتن با ensure_ascii=False برای نمایش صحیح فارسی
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()  # ذخیره آنی
                        stats["processed"] += 1
                        stats["cached"] += cached_count
                        stats["api_calls"] += sum(
                            1 for msg in result.get("messages", [])
                            if msg.get("content") not in STATIC_TRANSLATIONS.values()
                        )
                    else:
                        stats["errors"] += 1
                    
                    pbar.update(1)
                
                pbar.close()

    # نمایش آمار نهایی
    print(f"\n{'='*50}")
    print(f"✅ پایان کار!")
    print(f"📊 آمار:")
    print(f"   • خطوط پردازش‌شده: {stats['processed']}")
    print(f"   • خطوط با خطا: {stats['errors']}")
    print(f"   • متن‌های کش‌شده (بدون API): {stats['cached']}")
    print(f"   • درخواست‌های API ارسال‌شده: {stats['api_calls']}")
    print(f"🔑 کلیدهای فعال: {len(api_keys)}")
    print(f"💾 خروجی: {output_file}")
    print(f"{'='*50}")

# ---------------------------------------------------------
# اجرای اصلی
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "input_dataset.jsonl"
    OUTPUT_FILE = "output_dataset_fa.jsonl"
    
    # امکان تغییر مسیر از خط فرمان
    if len(sys.argv) >= 3:
        INPUT_FILE, OUTPUT_FILE = sys.argv[1], sys.argv[2]
    
    try:
        asyncio.run(main(INPUT_FILE, OUTPUT_FILE))
    except KeyboardInterrupt:
        print("\n⚠️ توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\n❌ خطای بحرانی: {e}")
        sys.exit(1)
