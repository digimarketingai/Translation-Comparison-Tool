"""
Translation Comparison Tool - Gradio Web Interface
翻譯比較工具 - Gradio 網頁介面

Run with: python app.py
執行方式: python app.py
"""

import os
import json
import time
import gradio as gr
import pandas as pd
from typing import Tuple, Optional

# Import the main module
try:
    from translation_compare import (
        TranslationComparer,
        GoogleTranslatorWrapper,
        MistralTranslator,
        SAMPLE_INPUTS,
        get_language_name
    )
except ImportError:
    print("⚠️ Could not import translation_compare module.")
    print("   Make sure translation_compare.py is in the same directory.")
    raise


# ============================================================
# GLOBAL INSTANCES
# ============================================================

comparer = TranslationComparer()


# ============================================================
# INTERFACE FUNCTIONS
# ============================================================

def initialize_mistral(api_key: str) -> str:
    """Initialize Mistral API with the provided key."""
    global comparer
    
    if api_key and api_key.strip():
        success = comparer.set_mistral_key(api_key.strip())
        if success:
            return "✅ Mistral API configured successfully! Both translators are ready. Mistral API 設定成功！兩種翻譯器都已準備就緒。"
        else:
            return "❌ Failed to initialize Mistral API. Please check your API key. 無法初始化 Mistral API，請檢查您的 API 金鑰。"
    else:
        return "⚠️ No API key provided. Only Google Translate will be available. 未提供 API 金鑰，僅能使用 Google 翻譯。"


def compare_translations(
    input_text: str,
    target_language: str,
    include_google: bool,
    include_mistral: bool
) -> Tuple[str, str, pd.DataFrame, str]:
    """
    Compare translations from Google Translate and Mistral AI.
    比較 Google 翻譯和 Mistral AI 的翻譯結果。
    """
    global comparer
    
    # Empty DataFrame for errors
    empty_df = pd.DataFrame(columns=["#", "Original 原文", "Google Translate", "Mistral AI", "Source Lang", "Match"])
    
    if not input_text or not input_text.strip():
        return "", "", empty_df, "⚠️ Please enter some text to translate. 請輸入要翻譯的文字。"
    
    # Parse input - one term/sentence per line
    lines = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
    
    if not lines:
        return "", "", empty_df, "⚠️ No valid text found. Enter one term per line. 未找到有效文字，請每行輸入一個詞彙。"
    
    # Map target language
    target_map = {
        "Chinese (Traditional) 繁體中文": "zh-TW",
        "Chinese (Simplified) 简体中文": "zh-CN",
        "English 英文": "en"
    }
    target = target_map.get(target_language, "en")
    
    status_messages = []
    
    # Check Mistral availability
    if include_mistral and not comparer.mistral.is_available():
        status_messages.append("⚠️ Mistral AI not configured. Please set API key. Mistral AI 未設定，請設定 API 金鑰。")
        include_mistral = False
    
    # Run comparison
    status_messages.append("🔄 Running translations... 正在翻譯...")
    
    try:
        results = comparer.compare(
            lines,
            target,
            use_google=include_google,
            use_mistral=include_mistral
        )
        
        if include_google:
            status_messages.append(f"✅ Google Translate: {len(results)} items translated Google 翻譯：已翻譯 {len(results)} 項")
        if include_mistral:
            status_messages.append(f"✅ Mistral AI: {len(results)} items translated Mistral AI：已翻譯 {len(results)} 項")
        
    except Exception as e:
        return "", "", empty_df, f"❌ Error 錯誤: {str(e)}"
    
    # Build table data
    table_data = []
    for i, r in enumerate(results, 1):
        google_trans = r.google_translation or "-"
        mistral_trans = r.mistral_translation or "-"
        
        table_data.append({
            "#": i,
            "Original 原文": r.original,
            "Google Translate": google_trans if include_google else "-",
            "Mistral AI": mistral_trans if include_mistral else "-",
            "Source Lang": get_language_name(r.source_lang),
            "Match": "✓" if r.translations_match else ("✗" if include_google and include_mistral else "-")
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Generate summary
    summary_lines = [
        "=" * 50,
        "📊 TRANSLATION COMPARISON SUMMARY 翻譯比較摘要",
        "=" * 50,
        f"Total items 總項目數: {len(lines)}",
        f"Target language 目標語言: {target_language}",
        "",
        "Results 結果:"
    ]
    
    for item in table_data:
        summary_lines.append(f"\n【{item['#']}】 {item['Original 原文']} ({item['Source Lang']})")
        if include_google:
            summary_lines.append(f"   🟢 Google: {item['Google Translate']}")
        if include_mistral:
            summary_lines.append(f"   🔵 Mistral: {item['Mistral AI']}")
    
    summary = "\n".join(summary_lines)
    
    # Generate HTML comparison
    html_rows = []
    for item in table_data:
        google_cell = item['Google Translate'] if include_google else "<span style='color: gray;'>-</span>"
        mistral_cell = item['Mistral AI'] if include_mistral else "<span style='color: gray;'>-</span>"
        
        # Highlight differences
        if include_google and include_mistral and google_cell != mistral_cell and google_cell != "-" and mistral_cell != "-":
            diff_style = "background-color: #fff3cd;"
        else:
            diff_style = ""
        
        match_icon = item['Match']
        match_color = "#28a745" if match_icon == "✓" else "#dc3545" if match_icon == "✗" else "#6c757d"
        
        html_rows.append(f"""
        <tr style="{diff_style}">
            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; font-weight: bold;">{item['#']}</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; font-weight: 500;">{item['Original 原文']}</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; background-color: #e8f5e9;">{google_cell}</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; background-color: #e3f2fd;">{mistral_cell}</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; color: #666;">{item['Source Lang']}</td>
            <td style="padding: 10px; border: 1px solid #dee2e6; text-align: center; color: {match_color}; font-size: 18px;">{match_icon}</td>
        </tr>
        """)
    
    html_table = f"""
    <style>
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans TC', sans-serif;
            font-size: 14px;
            margin: 10px 0;
        }}
        .comparison-table th {{
            padding: 12px 10px;
            text-align: left;
            border: 1px solid #dee2e6;
            font-weight: 600;
        }}
        .comparison-table tr:hover {{
            background-color: #f8f9fa !important;
        }}
    </style>
    <table class="comparison-table">
        <thead>
            <tr>
                <th style="width: 40px; background-color: #495057; color: white; text-align: center;">#</th>
                <th style="width: 22%; background-color: #495057; color: white;">Original 原文</th>
                <th style="width: 28%; background-color: #2e7d32; color: white;">🟢 Google Translate</th>
                <th style="width: 28%; background-color: #1565c0; color: white;">🔵 Mistral AI</th>
                <th style="width: 10%; background-color: #495057; color: white; text-align: center;">Lang</th>
                <th style="width: 5%; background-color: #495057; color: white; text-align: center;">Match</th>
            </tr>
        </thead>
        <tbody>
            {"".join(html_rows)}
        </tbody>
    </table>
    <p style="color: #666; font-size: 12px; margin-top: 10px;">
        💡 Yellow highlight indicates different translations between services. 黃色底色表示兩種翻譯結果不同。
    </p>
    """
    
    status = "\n".join(status_messages)
    
    return summary, html_table, df, status


def load_sample(sample_name: str) -> str:
    """Load sample input data."""
    return SAMPLE_INPUTS.get(sample_name, "")


def export_csv(df: pd.DataFrame) -> Optional[str]:
    """Export DataFrame to CSV file."""
    if df is None or df.empty:
        return None
    
    filename = f"translation_comparison_{int(time.time())}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filename


# ============================================================
# GRADIO INTERFACE
# ============================================================

def create_app():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="Translation Comparison Tool 翻譯比較工具",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .translator-badge { 
            display: inline-block; 
            padding: 4px 12px; 
            border-radius: 12px; 
            font-size: 12px; 
            font-weight: 500;
        }
        .google-badge { background-color: #e8f5e9; color: #2e7d32; }
        .mistral-badge { background-color: #e3f2fd; color: #1565c0; }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        <div class="main-header">
        
        # 🌐 Translation Comparison Tool 翻譯比較工具
        
        **Compare translations from Google Translate and Mistral AI side-by-side!**
        
        並排比較 Google 翻譯和 Mistral AI 的翻譯結果！
        
        </div>
        
        <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
            <span class="translator-badge google-badge">🟢 Google Translate - Free, No API Key 免費，無需 API</span>
            <span class="translator-badge mistral-badge">🔵 Mistral AI - Advanced AI Translation 進階 AI 翻譯</span>
        </div>
        """)
        
        # API Key Section
        with gr.Accordion("🔑 Mistral API Key (Optional - 可選)", open=False):
            gr.Markdown("""
            **Note 注意:** Google Translate works without any API key. Add Mistral API key for AI-powered comparison.
            
            Google 翻譯無需 API 金鑰即可使用。添加 Mistral API 金鑰以進行 AI 翻譯比較。
            """)
            with gr.Row():
                api_input = gr.Textbox(
                    label="Mistral API Key",
                    placeholder="Enter your Mistral API key... 輸入您的 Mistral API 金鑰...",
                    type="password",
                    scale=4
                )
                api_btn = gr.Button("Set API Key 設定金鑰", variant="primary", scale=1)
            api_status = gr.Textbox(
                label="Status 狀態",
                value="ℹ️ Google Translate is ready. Set Mistral API key for comparison. Google 翻譯已就緒，設定 Mistral API 金鑰以進行比較。",
                interactive=False
            )
        
        api_btn.click(initialize_mistral, inputs=[api_input], outputs=[api_status])
        
        gr.Markdown("---")
        
        # Main Input Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Text 輸入文字")
                
                input_text = gr.Textbox(
                    label="Enter terms or sentences (one per line) 輸入詞彙或句子（每行一個）",
                    placeholder="Enter text here...\nOne term or sentence per line\n每行輸入一個詞彙或句子\n\nExample 範例:\nhello\nworld\n你好",
                    lines=12,
                    max_lines=25
                )
                
                # Sample data buttons
                gr.Markdown("**📚 Load Sample Data 載入範例資料:**")
                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_INPUTS.keys()),
                        label="Select Sample 選擇範例",
                        value=None,
                        scale=3
                    )
                    load_btn = gr.Button("Load 載入", scale=1)
                
                load_btn.click(load_sample, inputs=[sample_dropdown], outputs=[input_text])
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings 設定")
                
                target_lang = gr.Radio(
                    choices=[
                        "Chinese (Traditional) 繁體中文",
                        "Chinese (Simplified) 简体中文",
                        "English 英文"
                    ],
                    label="Target Language 目標語言",
                    value="Chinese (Traditional) 繁體中文"
                )
                
                gr.Markdown("**Select Translators 選擇翻譯器:**")
                
                with gr.Row():
                    use_google = gr.Checkbox(
                        label="🟢 Google Translate",
                        value=True
                    )
                    use_mistral = gr.Checkbox(
                        label="🔵 Mistral AI",
                        value=True
                    )
                
                translate_btn = gr.Button(
                    "🚀 Compare Translations 比較翻譯",
                    variant="primary",
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="Processing Status 處理狀態",
                    lines=5,
                    interactive=False
                )
        
        gr.Markdown("---")
        
        # Results Section
        gr.Markdown("## 📊 Comparison Results 比較結果")
        
        with gr.Tabs():
            with gr.TabItem("📋 Visual Comparison 視覺比較"):
                html_output = gr.HTML(
                    label="Comparison Table",
                    value="""
                    <div style="text-align: center; padding: 60px; color: #888; background-color: #f8f9fa; border-radius: 8px;">
                        <p style="font-size: 18px;">📝 Enter text and click "Compare Translations" to see results</p>
                        <p>輸入文字並點擊「比較翻譯」查看結果</p>
                    </div>
                    """
                )
            
            with gr.TabItem("📝 Text Summary 文字摘要"):
                summary_output = gr.Textbox(
                    label="Translation Summary 翻譯摘要",
                    lines=20,
                    interactive=False
                )
            
            with gr.TabItem("📊 Data Table 資料表格"):
                df_output = gr.Dataframe(
                    label="Comparison Data 比較資料",
                    interactive=False,
                    wrap=True
                )
                
                with gr.Row():
                    export_btn = gr.Button("📥 Export CSV 匯出 CSV")
                    export_file = gr.File(label="Download 下載")
        
        # Connect translate button
        translate_btn.click(
            fn=compare_translations,
            inputs=[input_text, target_lang, use_google, use_mistral],
            outputs=[summary_output, html_output, df_output, status_output]
        )
        
        # Export CSV
        export_btn.click(
            fn=export_csv,
            inputs=[df_output],
            outputs=[export_file]
        )
        
        gr.Markdown("---")
        
        # Information Section
        with gr.Accordion("ℹ️ About This Tool 關於此工具", open=False):
            gr.Markdown("""
            ## 🔍 How It Works 工作原理
            
            1. **Enter Text 輸入文字**: Input terms or sentences, one per line 每行輸入一個詞彙或句子
            2. **Select Target 選擇目標**: Choose target language (Chinese or English) 選擇目標語言
            3. **Compare 比較**: Click to see translations from both services 點擊查看兩種翻譯服務的結果
            
            ## 🌐 Translators 翻譯器
            
            | Service 服務 | Features 特點 | API Key 金鑰 |
            |---------|----------|---------|
            | **Google Translate** | Fast, widely used, good for common phrases 快速、廣泛使用、適合常用語句 | ❌ Not needed 不需要 |
            | **Mistral AI** | Context-aware, better for nuanced text 具上下文意識、適合細膩文字 | ✅ Required 需要 |
            
            ## 💡 Tips 提示
            
            - **Yellow highlight 黃色底色**: Indicates different translations between services 表示兩種翻譯結果不同
            - **Match column**: ✓ = translations match, ✗ = translations differ
            - Use **Google Translate** for quick, everyday translations 快速日常翻譯
            - Use **Mistral AI** for technical, literary, or context-sensitive content 技術、文學或需要上下文的內容
            
            ## 🔤 Supported Languages 支援的語言
            
            Auto-detection works for: English, Chinese, Japanese, Korean, Spanish, French, German, Russian, Arabic, Thai, Vietnamese, and more.
            
            自動偵測支援：英文、中文、日文、韓文、西班牙文、法文、德文、俄文、阿拉伯文、泰文、越南文等。
            
            ## 📚 Sample Data Categories 範例資料類別
            
            - **Tech Terms**: AI, cloud computing, blockchain terminology
            - **Buddhist Terms**: Religious and philosophical terms
            - **Medical Terms**: Healthcare vocabulary
            - **Chinese Idioms**: Classical expressions (成語)
            - **Daily Phrases**: Common expressions
            - **Legal Terms**: Law terminology
            - **Financial Terms**: Finance vocabulary
            - **Mixed Languages**: Multi-language test
            """)
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🌐 Translation Comparison Tool 翻譯比較工具</p>
            <p style="font-size: 12px;">Google Translate (via deep-translator) & Mistral AI</p>
            <p style="font-size: 12px;">Made with ❤️ for language learners and translators</p>
        </div>
        """)
    
    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌐 Translation Comparison Tool 翻譯比較工具")
    print("=" * 60)
    print()
    print("🟢 Google Translate: Ready (no API key needed)")
    print("   Google 翻譯：已就緒（無需 API 金鑰）")
    print()
    print("🔵 Mistral AI: Set API key in the interface")
    print("   Mistral AI：請在介面中設定 API 金鑰")
    print()
    print("=" * 60)
    print()
    
    # Check for API key in environment
    if os.environ.get("MISTRAL_API_KEY"):
        comparer.set_mistral_key(os.environ.get("MISTRAL_API_KEY"))
        print("✅ Mistral API key loaded from environment variable.")
        print("   已從環境變數載入 Mistral API 金鑰。")
        print()
    
    demo = create_app()
    demo.launch(share=True)
