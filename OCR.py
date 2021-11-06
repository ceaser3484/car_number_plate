from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import cv2

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = "f9fb2f49a3ad459c8df20ec8ea913e29"
endpoint = "https://dongyeong-ocr.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

'''
OCR: Read File using the Read API, extract text - remote
This example will extract text in an image, then print results, line by line.
This API call can also extract handwriting style text (not shown).
'''

def paddingImg(img, targetSize):
    # 가로세로비 확인
    height, width, _ = img.shape

    # 패딩할 값 계산(가로세로비 유지)
    # 가로, 세로중 큰 값을 기준으로
    if width >= height:
        standard = width
        follower = height
    else:
        standard = height
        follower = width

    # 가로 세로 중 더 큰 값을 목표치에 맞췄을 때의 비율
    resizeRatio = targetSize / standard
    # 가로 세로 중 더 작은 값을 비율에 맞췄을 때의 크기
    followerSize = int(follower * resizeRatio)
    # 목표치와의 차이만큼 패딩필요한 값 계산, 양쪽에 넣기 때문에 /2
    required = (targetSize - followerSize) // 2
    requiredCompensation = required  # 보정치, 수평일 경우 왼쪽, 수직일 경우 위에 보정
    # 패딩이 정수값만 가능해서, 나누기로 인해 버려지는 값에 의해서 목표 사이즈에 비해 부족할 경우 값 보정
    if targetSize > (required * 2) + followerSize:
        requiredCompensation += 1

    # 이미지 리사이즈(비율 맞춰서)
    if width >= height:
        img = cv2.resize(img, (targetSize, followerSize))
        img = cv2.copyMakeBorder(img, requiredCompensation, required,  # 가로를 맞췄으니, 부족한 세로에 패딩
                                 0, 0, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    else:
        img = cv2.resize(img, (followerSize, targetSize))  # 세로를 맞췄으니, 부족한 가로에 패딩
        img = cv2.copyMakeBorder(img, 0, 0,
                                 requiredCompensation, required, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    return img

def callOCR():
    img = open("tempForOCR.png", 'rb')

    # Call API with img
    read_response = computervision_client.read_in_stream(img, language="ko", raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    result = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                # result.append((line.text, line.bounding_box))
                result.append(line.text)
    return result