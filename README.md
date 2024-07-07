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
python recommender_service.py --mode train --model_code autoencoder
```
``` 
# app server에서 inference
python app.py
```

[dataset]
- Steam-200k : [https://www.kaggle.com/datasets/tamber/steam-video-games]
- Amazon games Review : [https://nijianmo.github.io/amazon/index.html]
- ml-100k 

[Model]
- AutoEncoder
- EASE
- Autoencoder with side information 
- vae (추가 예정)


[특이사항]

