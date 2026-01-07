# 🌐 Translation Comparison Tool 翻譯比較工具

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Compare translations from Google Translate and Mistral AI side-by-side!**

並排比較 Google 翻譯和 Mistral AI 的翻譯結果！

---

## ✨ Features 功能

| Feature | Description |
|---------|-------------|
| 🟢 **Google Translate** | Free, no API key needed 免費，無需 API 金鑰 |
| 🔵 **Mistral AI** | Advanced AI-powered translation 進階 AI 翻譯 |
| 🔍 **Auto-detect** | Automatic source language detection 自動偵測來源語言 |
| 📊 **Side-by-side** | Visual comparison table 視覺化比較表格 |
| 💾 **Export** | Download results as CSV 匯出結果為 CSV |
| 🌏 **Bilingual** | English & Chinese interface 中英雙語介面 |

---

## 🚀 Quick Start 快速開始

### Option 1: Google Colab (Recommended 推薦)

```python
# Install dependencies 安裝依賴
!pip install deep-translator mistralai gradio pandas -q

# Clone repository 克隆儲存庫
!git clone https://github.com/digimarketingai/Translation-Comparison-Tool.git
%cd Translation-Comparison-Tool

# Run the app 執行應用
!python app.py
