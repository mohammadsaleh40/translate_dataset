import os
import sys
import json
import asyncio
import aiohttp
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import time
from datetime import date

load_dotenv()

# ---------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-31b-it")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT", "30"))  # تعداد کل درخواست‌های همزمان
RATE_LIMIT_PER_MINUTE = 15
DAILY_QUOTA_PER_KEY = 1500

STATIC_TRANSLATIONS = {
    "You are a helpful AI assistant.": "تو یک دستیار هوش مصنوعی هستی."
}
_runtime_cache: dict[str, str] = {}

# ---------------------------------------------------------
# مدیریت کلیدها - بدون قفل سراسری!
# ---------------------------------------------------------
class KeyManager:
    def __init__(self, api_keys: list[str]):
        self.keys = api_keys
        self.lock = asyncio.Lock()  # فقط برای آپدیت اتمیک شمارنده‌ها
        
        self.minute_requests: dict[str, list[float]] = defaultdict(list)
        self.daily_counts: dict[str, int] = defaultdict(int)
        self.last_reset_date: str = date.today().isoformat()
        
        # برای لاگ‌گیری
        self.key_usage = {k: 0 for k in api_keys}

    def _reset_daily_if_needed(self):
        today = date.today().isoformat()
        if self.last_reset_date != today:
            self.daily_counts.clear()
            self.last_reset_date = today

    async def acquire_key(self) -> str:
        """
        دریافت یک کلید آزاد - قفل فقط برای چند میکروثانیه!
        """
        while True:
            self._reset_daily_if_needed()
            now = time.time()
            
            # کپی سریع از وضعیت کلیدها (بدون قفل طولانی)
            available = []
            async with self.lock:
                for key in self.keys:
                    # پاک‌سازی تایم‌استمپ‌های قدیمی
                    self.minute_requests[key] = [
                        t for t in self.minute_requests[key] if now - t < 60
                    ]
                    minute_ok = len(self.minute_requests[key]) < RATE_LIMIT_PER_MINUTE
                    daily_ok = self.daily_counts[key] < DAILY_QUOTA_PER_KEY
                    if minute_ok and daily_ok:
                        available.append(key)
            
            if available:
                # انتخاب کلید با کمترین استفاده (Load Balancing ساده)
                key = min(available, key=lambda k: self.key_usage[k])
                async with self.lock:
                    self.minute_requests[key].append(now)
                    self.daily_counts[key] += 1
                    self.key_usage[key] += 1
                return key
            
            # اگر هیچ کلیدی آزاد نبود، صبر کوتاه
            await asyncio.sleep(0.5)

# ---------------------------------------------------------
# ترجمه با Gemini API
# ---------------------------------------------------------
async def translate_with_gemma(
    session: aiohttp.ClientSession,
    text: str,
    api_key: str,
    key_id: str
) -> str:
    if not text or not text.strip():
        return text
    if text in _runtime_cache:
        return _runtime_cache[text]
    if text in STATIC_TRANSLATIONS:
        return STATIC_TRANSLATIONS[text]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    
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
                result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                _runtime_cache[text] = result
                return result
            
            elif response.status == 429:
                print(f"\n⚠️ Rate limit کلید {key_id}")
                return None
            elif response.status >= 500:
                print(f"\n⚠️ خطای سرور {response.status}")
                return None
            else:
                error = await response.text()
                print(f"\n❌ خطای API {response.status}: {error[:100]}")
                return text

    except asyncio.TimeoutError:
        print(f"\n⏱️ Timeout برای کلید {key_id}")
        return None
    except Exception as e:
        print(f"\n❌ خطا: {type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------
# ترجمه یک متن واحد (wrapper با مدیریت کلید)
# ---------------------------------------------------------
async def translate_single_text(
    session: aiohttp.ClientSession,
    key_manager: KeyManager,
    text: str,
    semaphore: asyncio.Semaphore
) -> str:
    
    # کش‌ها را قبل از گرفتن کلید چک کن
    if text in STATIC_TRANSLATIONS:
        return STATIC_TRANSLATIONS[text]
    if text in _runtime_cache:
        return _runtime_cache[text]
    
    async with semaphore:  # محدود کردن همزمانی کلی
        api_key = await key_manager.acquire_key()
        key_id = api_key[:12] + "..."
        result = await translate_with_gemma(session, text, api_key, key_id)
        return result if result else text  # fallback

# ---------------------------------------------------------
# پردازش یک خط با ترجمه موازی تمام contentها
# ---------------------------------------------------------
async def process_line_parallel(
    line: str,
    session: aiohttp.ClientSession,
    key_manager: KeyManager,
    semaphore: asyncio.Semaphore
) -> dict | None:
    
    try:
        data = json.loads(line.strip())
        messages = data.get("messages", [])
        
        # جمع‌آوری متن‌هایی که نیاز به ترجمه واقعی دارند
        items_to_translate = []  # [(msg_ref, original_text), ...]
        
        for msg in messages:
            if "content" not in msg or not isinstance(msg["content"], str):
                continue
            original = msg["content"]
            if original in STATIC_TRANSLATIONS:
                msg["content"] = STATIC_TRANSLATIONS[original]
            elif original in _runtime_cache:
                msg["content"] = _runtime_cache[original]
            else:
                items_to_translate.append((msg, original))
        
        if not items_to_translate:
            return data
        
        # 🚀 ترجمه موازی تمام متن‌های این خط با asyncio.gather
        tasks = [
            translate_single_text(session, key_manager, text, semaphore)
            for _, text in items_to_translate
        ]
        results = await asyncio.gather(*tasks)
        
        # اعمال نتایج
        for (msg, _), translated in zip(items_to_translate, results):
            msg["content"] = translated
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"⚠️ خطای JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ خطا: {type(e).__name__}: {e}")
        return None

# ---------------------------------------------------------
# تابع اصلی با پردازش موازی خطوط
# ---------------------------------------------------------
async def main(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"❌ فایل ورودی یافت نشد: {input_path}")

    # استخراج کلیدها
    api_keys = []
    for i in range(1, 21):
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())
    
    if not api_keys:
        raise ValueError("❌ هیچ کلید API معتبری یافت نشد.")
    
    print(f"🔑 {len(api_keys)} کلید بارگذاری شد")
    print(f"🤖 مدل: {MODEL_NAME}")
    print(f"⚡ حداکثر درخواست همزمان: {MAX_CONCURRENT_REQUESTS}")
    print(f"📊 محدودیت: {RATE_LIMIT_PER_MINUTE}/دقیقه، {DAILY_QUOTA_PER_KEY}/روز per key\n")

    key_manager = KeyManager(api_keys)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # کنترل همزمانی کلی
    
    stats = {"processed": 0, "errors": 0, "cached": 0, "api_calls": 0}
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        with open(output_file, "w", encoding="utf-8") as fout:
            total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
            
            with open(input_file, "r", encoding="utf-8") as fin:
                pbar = tqdm(desc="🔄 ترجمه", total=total_lines, unit="line")
                
                for line in fin:
                    if not line.strip():
                        pbar.update(1)
                        continue
                    
                    # آمار کش قبل از پردازش
                    data_pre = json.loads(line.strip())
                    cached_count = sum(
                        1 for msg in data_pre.get("messages", [])
                        if msg.get("content") in STATIC_TRANSLATIONS or msg.get("content") in _runtime_cache
                    )
                    
                    result = await process_line_parallel(line, session, key_manager, semaphore)
                    
                    if result:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        stats["processed"] += 1
                        stats["cached"] += cached_count
                        # تخمین API calls
                        msg_count = len([m for m in result.get("messages", []) if m.get("content") not in STATIC_TRANSLATIONS.values()])
                        stats["api_calls"] += msg_count
                    else:
                        stats["errors"] += 1
                    
                    # لاگ سرعت هر ۱۰ خط
                    if stats["processed"] % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = stats["processed"] / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({"سرعت": f"{rate:.1f} خط/ثانیه"})
                    
                    pbar.update(1)
                
                pbar.close()

    # آمار نهایی
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ پایان کار! زمان کل: {elapsed:.1f} ثانیه ({elapsed/60:.1f} دقیقه)")
    print(f"📊 آمار:")
    print(f"   • خطوط پردازش‌شده: {stats['processed']}")
    print(f"   • خطوط با خطا: {stats['errors']}")
    print(f"   • متن‌های کش‌شده: {stats['cached']}")
    print(f"   • درخواست‌های API: ~{stats['api_calls']}")
    print(f"   • میانگین سرعت: {stats['processed']/elapsed:.2f} خط/ثانیه")
    print(f"🔑 توزیع استفاده از کلیدها:")
    for key, count in key_manager.key_usage.items():
        print(f"   • {key[:15]}... : {count} درخواست")
    print(f"💾 خروجی: {output_file}")
    print(f"{'='*60}")

# ---------------------------------------------------------
# اجرا
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT = "input_dataset.jsonl"
    OUTPUT = "output_dataset_fa.jsonl"
    
    if len(sys.argv) >= 3:
        INPUT, OUTPUT = sys.argv[1], sys.argv[2]
    
    try:
        asyncio.run(main(INPUT, OUTPUT))
    except KeyboardInterrupt:
        print("\n⚠️ متوقف شد توسط کاربر")
    except Exception as e:
        print(f"\n❌ خطای بحرانی: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
