"""
6주차 모델 설정 — HuggingFace Inference API
===========================================
토큰은 .env 의 HF_TOKEN 또는 HUGGINGFACEHUB_API_TOKEN 에서 읽는다.
HF Space에 배포할 때는 Settings > Secrets 에 HF_TOKEN 을 등록한다.
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

# 음식 이미지 분류 (ViT, food-101 파인튜닝)
VISION_MODEL = "nateraw/food"

# 칼로리/영양소 추정용 텍스트 LLM
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN(또는 HUGGINGFACEHUB_API_TOKEN) 환경변수가 비어 있습니다.\n"
            "  1) https://huggingface.co/settings/tokens 에서 Read 토큰 발급\n"
            "  2) .env 에 HF_TOKEN=hf_xxx 추가 (로컬)\n"
            "  3) HF Space: Settings > Secrets 에 HF_TOKEN 등록"
        )
    return token


def get_client() -> InferenceClient:
    """이미지 분류용 클라이언트."""
    return InferenceClient(token=get_token())
