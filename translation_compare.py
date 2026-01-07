"""
Translation Comparison Tool - Google Translate vs Mistral AI
翻譯比較工具 - 谷歌翻譯 vs Mistral AI

Compare translations from Google Translate (free, no API key) and Mistral AI side-by-side.
並排比較 Google 翻譯（免費，無需 API）和 Mistral AI 的翻譯結果。

Author: DigiMarketingAI
GitHub: https://github.com/digimarketingai

Features 功能:
- Enter multiple terms/sentences (one per line) 輸入多個詞彙/句子（每行一個）
- Auto-detect source language 自動偵測來源語言
- Translate to Chinese or English 翻譯成中文或英文
- Compare Google Translate vs Mistral AI 比較兩種翻譯服務
- Export results to CSV 匯出結果為 CSV

Run with: python translation_compare.py
"""

import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================
# DEPENDENCY CHECKS
# ============================================================

try:
    from deep_translator import GoogleTranslator
except ImportError:
    raise ImportError(
        "Please install deep-translator: pip install deep-translator\n"
        "請安裝 deep-translator: pip install deep-translator"
    )

try:
    from mistralai import Mistral
except ImportError:
    raise ImportError(
        "Please install mistralai: pip install mistralai\n"
        "請安裝 mistralai: pip install mistralai"
    )


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class TranslatorType(Enum):
    """Translator service type 翻譯服務類型"""
    GOOGLE = "google"
    MISTRAL = "mistral"


class TargetLanguage(Enum):
    """Target language options 目標語言選項"""
    CHINESE_TRADITIONAL = "zh-TW"
    CHINESE_SIMPLIFIED = "zh-CN"
    ENGLISH = "en"


@dataclass
class TranslationResult:
    """Result of a single translation 單一翻譯結果"""
    original: str
    translation: str
    source_lang: str
    translator: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison of translations from multiple services 多服務翻譯比較結果"""
    original: str
    source_lang: str
    google_translation: Optional[str]
    mistral_translation: Optional[str]
    translations_match: bool


# ============================================================
# LANGUAGE UTILITIES
# ============================================================

LANGUAGE_NAMES = {
    'en': 'English',
    'zh': 'Chinese',
    'zh-CN': 'Chinese (Simplified)',
    'zh-TW': 'Chinese (Traditional)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian',
    'ar': 'Arabic',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'hi': 'Hindi',
    'id': 'Indonesian',
    'ms': 'Malay',
}


def detect_language_by_chars(text: str) -> str:
    """
    Detect language based on character types.
    根據字符類型偵測語言。
    """
    # Chinese characters
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'zh'
    # Japanese (Hiragana/Katakana)
    elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
        return 'ja'
    # Korean (Hangul)
    elif any('\uac00' <= char <= '\ud7af' for char in text):
        return 'ko'
    # Thai
    elif any('\u0e00' <= char <= '\u0e7f' for char in text):
        return 'th'
    # Arabic
    elif any('\u0600' <= char <= '\u06ff' for char in text):
        return 'ar'
    # Russian (Cyrillic)
    elif any('\u0400' <= char <= '\u04ff' for char in text):
        return 'ru'
    # Hindi (Devanagari)
    elif any('\u0900' <= char <= '\u097f' for char in text):
        return 'hi'
    # Default to English
    else:
        return 'en'


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    return LANGUAGE_NAMES.get(code, code)


# ============================================================
# GOOGLE TRANSLATOR
# ============================================================

class GoogleTranslatorWrapper:
    """
    Wrapper for Google Translate using deep-translator (no API key needed).
    使用 deep-translator 的 Google 翻譯包裝器（無需 API 金鑰）。
    """
    
    def __init__(self):
        self.name = "Google Translate"
        self.type = TranslatorType.GOOGLE
    
    def translate(self, text: str, target_lang: str) -> TranslationResult:
        """
        Translate single text to target language.
        將單一文本翻譯成目標語言。
        """
        try:
            # Clean text
            text = text.strip()
            if not text:
                return TranslationResult(
                    original=text,
                    translation="",
                    source_lang="",
                    translator=self.name,
                    success=False,
                    error_message="Empty text"
                )
            
            # Map language codes
            lang_map = {
                'zh': 'zh-CN',
                'zh-TW': 'zh-TW',
                'zh-CN': 'zh-CN',
                'en': 'en',
                'chinese': 'zh-CN',
                'english': 'en'
            }
            target = lang_map.get(target_lang, target_lang)
            
            # Detect source language
            source_lang = detect_language_by_chars(text)
            
            # Translate
            translator = GoogleTranslator(source='auto', target=target)
            result = translator.translate(text)
            
            return TranslationResult(
                original=text,
                translation=result,
                source_lang=source_lang,
                translator=self.name,
                success=True
            )
            
        except Exception as e:
            return TranslationResult(
                original=text,
                translation="",
                source_lang="unknown",
                translator=self.name,
                success=False,
                error_message=str(e)
            )
    
    def translate_batch(self, texts: List[str], target_lang: str) -> List[TranslationResult]:
        """
        Translate multiple texts.
        翻譯多個文本。
        """
        results = []
        for text in texts:
            if text.strip():
                result = self.translate(text.strip(), target_lang)
                results.append(result)
        return results


# ============================================================
# MISTRAL TRANSLATOR
# ============================================================

class MistralTranslator:
    """
    Translator using Mistral AI API.
    使用 Mistral AI API 的翻譯器。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.client = None
        self.name = "Mistral AI"
        self.type = TranslatorType.MISTRAL
        self.model = "mistral-small-latest"
        
        if self.api_key:
            try:
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ Failed to initialize Mistral client: {e}")
    
    def is_available(self) -> bool:
        """Check if Mistral client is available."""
        return self.client is not None
    
    def translate(self, text: str, target_lang: str) -> TranslationResult:
        """
        Translate single text using Mistral AI.
        使用 Mistral AI 翻譯單一文本。
        """
        if not self.client:
            return TranslationResult(
                original=text,
                translation="",
                source_lang="unknown",
                translator=self.name,
                success=False,
                error_message="Mistral API not configured"
            )
        
        try:
            text = text.strip()
            if not text:
                return TranslationResult(
                    original=text,
                    translation="",
                    source_lang="",
                    translator=self.name,
                    success=False,
                    error_message="Empty text"
                )
            
            # Map target language
            target_full = "Traditional Chinese (繁體中文)" if target_lang in ['zh', 'zh-TW'] else \
                          "Simplified Chinese (简体中文)" if target_lang == 'zh-CN' else \
                          "English"
            
            prompt = f"""Translate the following text to {target_full}. 
Provide ONLY the translation, no explanations.

Text: {text}

Translation:"""

            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            translation = response.choices[0].message.content.strip()
            source_lang = detect_language_by_chars(text)
            
            return TranslationResult(
                original=text,
                translation=translation,
                source_lang=source_lang,
                translator=self.name,
                success=True
            )
            
        except Exception as e:
            return TranslationResult(
                original=text,
                translation="",
                source_lang="unknown",
                translator=self.name,
                success=False,
                error_message=str(e)
            )
    
    def translate_batch(self, texts: List[str], target_lang: str) -> List[TranslationResult]:
        """
        Translate multiple texts using batch API call.
        使用批次 API 呼叫翻譯多個文本。
        """
        if not self.client:
            return [
                TranslationResult(
                    original=t,
                    translation="",
                    source_lang="unknown",
                    translator=self.name,
                    success=False,
                    error_message="Mistral API not configured"
                ) for t in texts
            ]
        
        # Clean texts
        clean_texts = [t.strip() for t in texts if t.strip()]
        if not clean_texts:
            return []
        
        # Map target language
        target_full = "Traditional Chinese (繁體中文)" if target_lang in ['zh', 'zh-TW'] else \
                      "Simplified Chinese (简体中文)" if target_lang == 'zh-CN' else \
                      "English"
        
        texts_numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(clean_texts)])
        
        prompt = f"""You are a professional translator. Translate the following texts to {target_full}.

Auto-detect the source language for each text. Provide natural, contextually appropriate translations.

Texts to translate:
{texts_numbered}

Respond in this exact JSON format only:
{{
    "translations": [
        {{"original": "text1", "translation": "translated text1", "source_lang": "en"}},
        {{"original": "text2", "translation": "translated text2", "source_lang": "zh"}}
    ]
}}

Use language codes: en, zh, ja, ko, es, fr, de, ru, ar, th, etc.
Respond with JSON only, no additional text."""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.choices[0].message.content
            
            # Parse JSON
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                data = json.loads(match.group())
                translations = data.get("translations", [])
                
                results = []
                for item in translations:
                    results.append(TranslationResult(
                        original=item.get("original", ""),
                        translation=item.get("translation", ""),
                        source_lang=item.get("source_lang", "unknown"),
                        translator=self.name,
                        success=True
                    ))
                return results
            
        except Exception as e:
            return [
                TranslationResult(
                    original=t,
                    translation="",
                    source_lang="unknown",
                    translator=self.name,
                    success=False,
                    error_message=str(e)
                ) for t in clean_texts
            ]
        
        return [
            TranslationResult(
                original=t,
                translation="",
                source_lang="unknown",
                translator=self.name,
                success=False,
                error_message="Parse error"
            ) for t in clean_texts
        ]


# ============================================================
# COMPARISON ENGINE
# ============================================================

class TranslationComparer:
    """
    Compare translations from multiple services.
    比較多個翻譯服務的結果。
    """
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.google = GoogleTranslatorWrapper()
        self.mistral = MistralTranslator(api_key=mistral_api_key)
    
    def set_mistral_key(self, api_key: str) -> bool:
        """Set or update Mistral API key."""
        self.mistral = MistralTranslator(api_key=api_key)
        return self.mistral.is_available()
    
    def compare(
        self,
        texts: List[str],
        target_lang: str,
        use_google: bool = True,
        use_mistral: bool = True
    ) -> List[ComparisonResult]:
        """
        Compare translations from selected services.
        比較所選服務的翻譯結果。
        """
        results = []
        
        # Clean texts
        clean_texts = [t.strip() for t in texts if t.strip()]
        if not clean_texts:
            return results
        
        # Get Google translations
        google_results = {}
        if use_google:
            for result in self.google.translate_batch(clean_texts, target_lang):
                if result.success:
                    google_results[result.original] = result.translation
        
        # Get Mistral translations
        mistral_results = {}
        if use_mistral and self.mistral.is_available():
            for result in self.mistral.translate_batch(clean_texts, target_lang):
                if result.success:
                    mistral_results[result.original] = result.translation
        
        # Build comparison results
        for text in clean_texts:
            google_trans = google_results.get(text)
            mistral_trans = mistral_results.get(text)
            
            # Check if translations match (ignore case and whitespace)
            match = False
            if google_trans and mistral_trans:
                match = google_trans.strip().lower() == mistral_trans.strip().lower()
            
            results.append(ComparisonResult(
                original=text,
                source_lang=detect_language_by_chars(text),
                google_translation=google_trans,
                mistral_translation=mistral_trans,
                translations_match=match
            ))
        
        return results
    
    def compare_to_dict(self, results: List[ComparisonResult]) -> List[Dict]:
        """Convert comparison results to list of dictionaries."""
        return [
            {
                "original": r.original,
                "source_lang": get_language_name(r.source_lang),
                "google_translation": r.google_translation or "-",
                "mistral_translation": r.mistral_translation or "-",
                "match": "✓" if r.translations_match else "✗"
            }
            for r in results
        ]
    
    def compare_to_json(self, results: List[ComparisonResult], indent: int = 2) -> str:
        """Convert comparison results to JSON string."""
        return json.dumps(self.compare_to_dict(results), ensure_ascii=False, indent=indent)


# ============================================================
# SAMPLE DATA
# ============================================================

SAMPLE_INPUTS = {
    "Tech Terms (EN→ZH)": """artificial intelligence
machine learning
cloud computing
blockchain
cybersecurity
virtual reality
augmented reality
internet of things
big data
quantum computing""",

    "Buddhist Terms (EN→ZH)": """Buddha
Dharma
Sangha
Nirvana
Karma
Meditation
Enlightenment
Sutra
Bodhisattva
Zen""",

    "Medical Terms (EN→ZH)": """diagnosis
symptom
treatment
vaccine
antibody
virus
bacteria
inflammation
chronic
acute""",

    "Chinese Idioms (ZH→EN)": """一石二鳥
畫龍點睛
對牛彈琴
井底之蛙
守株待兔
塞翁失馬
三思而後行
百聞不如一見
入鄉隨俗
熟能生巧""",

    "Daily Phrases (ZH→EN)": """你好
謝謝
不客氣
早安
晚安
再見
請問
對不起
沒問題
加油""",

    "Legal Terms (EN→ZH)": """jurisdiction
plaintiff
defendant
litigation
arbitration
injunction
indemnity
liability
precedent
statute""",

    "Financial Terms (EN→ZH)": """equity
dividend
portfolio
hedge fund
derivatives
liquidity
amortization
collateral
yield
volatility""",

    "Mixed Languages": """Hello World
你好世界
Bonjour le monde
Hola Mundo
こんにちは世界
안녕하세요 세계
Привет мир
مرحبا بالعالم"""
}


def get_sample_names() -> List[str]:
    """Get list of available sample names."""
    return list(SAMPLE_INPUTS.keys())


def get_sample_text(name: str) -> str:
    """Get sample text by name."""
    return SAMPLE_INPUTS.get(name, "")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_compare(
    text: str,
    target_lang: str = "zh-TW",
    mistral_api_key: Optional[str] = None
) -> List[Dict]:
    """
    Quick comparison of a single text or multiple texts (newline separated).
    快速比較單一或多個文本（換行分隔）。
    """
    texts = [t.strip() for t in text.strip().split('\n') if t.strip()]
    comparer = TranslationComparer(mistral_api_key)
    results = comparer.compare(texts, target_lang)
    return comparer.compare_to_dict(results)


def translate_google(text: str, target_lang: str = "zh-TW") -> str:
    """Quick Google Translate."""
    translator = GoogleTranslatorWrapper()
    result = translator.translate(text, target_lang)
    return result.translation if result.success else f"Error: {result.error_message}"


def translate_mistral(text: str, target_lang: str = "zh-TW", api_key: Optional[str] = None) -> str:
    """Quick Mistral AI Translate."""
    translator = MistralTranslator(api_key)
    result = translator.translate(text, target_lang)
    return result.translation if result.success else f"Error: {result.error_message}"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌐 Translation Comparison Tool - 翻譯比較工具")
    print("=" * 60)
    print()
    print("This module provides translation comparison functionality.")
    print("此模組提供翻譯比較功能。")
    print()
    print("Usage 使用方式:")
    print()
    print("  from translation_compare import TranslationComparer")
    print("  comparer = TranslationComparer(mistral_api_key='your-key')")
    print("  results = comparer.compare(['hello', 'world'], 'zh-TW')")
    print()
    print("Or quick compare 或快速比較:")
    print()
    print("  from translation_compare import quick_compare")
    print("  results = quick_compare('hello\\nworld', 'zh-TW')")
    print()
    print("Run 'python app.py' for the web interface.")
    print("執行 'python app.py' 啟動網頁介面。")
    print("=" * 60)
