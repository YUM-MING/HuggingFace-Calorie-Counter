---
title: HuggingFace Calorie Counter
emoji: 🍱
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
---

# 6주차: HuggingFace Space 칼로리 카운터

음식 사진을 업로드하면 HuggingFace Inference API로 음식을 인식하고,
LLM이 1인분 기준 칼로리/탄단지를 추정하는 Gradio 앱이다.
완성된 앱은 그대로 **HuggingFace Space** 에 올려 공개 URL을 받을 수 있다.

> 이 자료의 기술 정보는 2026-04 기준입니다.

## 파일 구조

```
week06/
├── app.py              # Gradio 앱 본체 (TODO 4곳을 채워 완성)
├── model_config.py     # HF 모델 이름 상수 + InferenceClient 팩토리
├── requirements.txt    # Space가 설치할 의존성
├── .env.example        # HF_TOKEN 템플릿
└── README.md           # 이 파일 (앞의 YAML이 Space 설정)
```

## 실행 방법 (로컬)

```bash
# 1) 토큰 발급: https://huggingface.co/settings/tokens (Read 권한)
cp .env.example .env
# .env 파일을 열어 HF_TOKEN=hf_... 로 수정

# 2) 의존성 설치
uv pip install -r requirements.txt

# 3) 실행
uv run python app.py
# 브라우저: http://127.0.0.1:7860
```

## TODO (app.py 안)

1. **`SYSTEM_PROMPT`** — 영양사 역할 + JSON 스키마 강제 (ChatPromptTemplate 안에 들어가므로 중괄호는 `{{ }}`로 이스케이프)
2. **`_chain_lazy()` 안 LCEL 체인 조립** (4단계)
   - 2-1. `HuggingFaceEndpoint` 생성 (`repo_id=LLM_MODEL`, `task="text-generation"`, 토큰)
   - 2-2. `ChatHuggingFace(llm=endpoint)` 로 감싸기
   - 2-3. `ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "...{labels_json}")])`
   - 2-4. `_chain = prompt | llm | JsonOutputParser()` 파이프 연결
3. **`classify_food()`** — `client.image_classification(tmp_path, model=VISION_MODEL)` 호출
4. **`estimate_calories()`** — `chain.invoke({"labels_json": labels_json})` 호출

네 곳을 채우면 앱이 동작한다.

## HuggingFace Space 배포 방법

1. https://huggingface.co/new-space 에서 **Gradio** SDK로 Space 생성
2. Space의 **Settings > Variables and secrets** 에 `HF_TOKEN` 등록
3. 파일 업로드 (웹 드래그 or `git push`)
   ```bash
   git clone https://huggingface.co/spaces/<본인>/<space-이름>
   cd <space-이름>
   cp ../week06/{app.py,model_config.py,requirements.txt,README.md} .
   git add .
   git commit -m "init"
   git push
   ```
4. 몇 분 후 `https://huggingface.co/spaces/<본인>/<space-이름>` 접속하여 동작 확인

## 과제

- [ ] Space URL 제출
- [ ] title/emoji/프롬프트 중 하나 이상 커스터마이즈
- [ ] 결과 스크린샷 1장 + 2~3줄 설명

## 사용 모델

| 역할 | 모델 |
|------|------|
| 이미지 분류 | `nateraw/food` |
| 칼로리 추정 LLM | `meta-llama/Meta-Llama-3-8B-Instruct` |
