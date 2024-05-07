# recommendation_pipeline
여러 개의 데이터셋과 모델 파이프라인을 테스트하는 코드
성능을 높이기 위한 Meta 정보 사용은 자명하지만...
interaction 구성으로 기본적인 시스템을 구축하고 내부 로직을 확장성있게 가져가려고 한다.

[dataset]
- Steam-200k : [https://www.kaggle.com/datasets/tamber/steam-video-games]
- Amazon games Review : [https://nijianmo.github.io/amazon/index.html]

[Model]
- AutoEncoder
- EASE (추가 예정)

[특이사항]
- hyper parameter config 관리 방안을 argparse를 사용할 것은 검토 중

