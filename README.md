# recommendation_pipeline
여러 개의 데이터셋과 모델 파이프라인을 테스트하는 코드

[Command]
``` 
# 가상 환경 구축 
python -m venv .venv
```
```
# poetry 설치
poetry install
``` 
``` 
# model 학습
python recommender_service.py 
```
```
# gradio에서 demo
python demo.py
```
``` 
# app server에서 inference
python app.py
```

[dataset]
- Steam-200k : [https://www.kaggle.com/datasets/tamber/steam-video-games]
- Amazon games Review : [https://nijianmo.github.io/amazon/index.html]
- ml-100k : [https://www.kaggle.com/datasets/abhikjha/movielens-100k]

[Model]
- AutoEncoder
- EASE
- Autoencoder with side information 
- vae (추가 예정)

[MLConfiguration]
- hydra : 파라미터 관리, 동적 실험

[MLOps]
- Bentoml : 모델 패키징 & 서빙
- MLFlow : Ops 시스템 구축

[Service]
- docker : 컨테이너 환경 구축 
- fastapi : 추천 API 통신

[PoC]
- gradio : PoC 데모 

