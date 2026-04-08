# 🍱 AI Foodie: Image-Based Calorie Counter
**HuggingFace Inference API & LangChain LCEL 기반 인공지능 영양사 서비스**

https://huggingface.co/spaces/eunanim/calorie-counter/

음식 사진을 업로드하면 이미지 분류 모델이 음식을 식별하고, LLM이 1인분 기준 영양 성분(칼로리, 탄단지)을 추정하여 제공합니다.

---

## 🚀 Key Features
* **Computer Vision 기반 음식 인식**: HuggingFace의 최첨단 이미지 분류 모델을 사용하여 이미지 내 음식을 분석합니다.
* **LangChain LCEL 체인 설계**: `PromptTemplate | LLM | JsonOutputParser` 구조의 선언적 체인을 활용하여 안정적인 JSON 응답을 보장합니다.
* **실시간 영양 정보 추정**: 영양사 AI 페르소나를 탑재한 LLM이 식재료의 비율과 조리법을 고려하여 영양 정보를 도출합니다.
* **Gradio UI**: 사용자 친화적인 웹 인터페이스를 통해 로컬 및 HuggingFace Space에서 즉시 사용 가능합니다.

---

## 🛠 Tech Stack
* **Language**: ![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
* **LLM Framework**: ![LangChain](https://img.shields.io/badge/LangChain-LCEL-blue?logo=chainlink)
* **Model Hosting**: ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-FFD21E)
* **Web UI**: ![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
* **Package Manager**: ![uv](https://img.shields.io/badge/uv-Package%20Manager-purple)

---

## 📂 Project Structure
```text
├── app.py             # Gradio 애플리케이션 및 LCEL 체인 로직
├── model_config.py    # 모델 설정 및 API 토큰 관리
├── .env               # 환경 변수 (HF_TOKEN 등)
└── requirements.txt   # 의존성 패키지 목록
