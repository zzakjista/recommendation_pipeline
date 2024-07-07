from recommender_service import inference

def get_recommendations(request):
    """
    request를 받아 response를 반환하는 함수
    :input: 
        - request
    :return:
        - response
    """
    response = inference(0)
    return response