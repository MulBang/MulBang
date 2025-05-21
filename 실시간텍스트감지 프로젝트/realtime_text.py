from imutils.video import VideoStream
from imutils.video import FPS 
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2
import pytesseract
import multiprocessing
from PIL import ImageFont, ImageDraw, Image

# 텍스트 OCR을 위한 pytesseract 경로
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

prevTime = 0 # 이전 시간을 저장할 변수

def box_extractor(scores, geometry, min_confidence):
    # 찾았던 scores, geometry을 루프로 돌며 min_confidence 이상의 값들만 모아서
    # rectangles 배열과 confidence값으로 반환
    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences


def process_detection(roi):
    # tesseract OCR 엔진 옵션
    # 언어 : 영어, 한글, 기본 + LSTM 엔진 사용
    config = '-l eng+kor --oem 3 --psm 7'
    text = pytesseract.image_to_string(roi[0], config=config)

    return text, roi[1]


if __name__ == '__main__':
    # 구동 옵션, 글자 위치 찾는데 필요한 학습 파일과 confidence, 화면 크기, 패딩을 설정
    args = {
        'east': 'frozen_east_text_detection.pb',
        'min_confidence': 0.7,
        'width': 320,
        'height': 320,
        'padding': 0.0,
    }

    w, h = None, None
    new_w, new_h = args['width'], args['height']
    ratio_w, ratio_h = None, None

    # 글자 위치를 파악할 수 있는 레이어 구성
    layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    # opencv 모듈에서 신경망 모델 미리 로드
    net = cv2.dnn.readNet(args["east"])
    
    # 웹캠 구동
    vs = VideoStream(src=0).start()
    time.sleep(1)

    fps = FPS().start()

    # 폰트 설정
    font = ImageFont.truetype('fonts/gulim.ttc', 30)
    
    # 메인루프
    while True:

        # 프레임 읽기
        frame = vs.read()

        if frame is None:
            break

        # 프레임 리사이즈
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        
        # opencv는 한글이 깨져서 한글을 화면에 출력하기 위해 PIL module을 이용
        img = Image.fromarray(orig)
        draw = ImageDraw.Draw(img)
        orig_h, orig_w = orig.shape[:2]

        if w is None or h is None:
            h, w = frame.shape[:2]
            ratio_w = w / float(new_w)
            ratio_h = h / float(new_h)

        frame = cv2.resize(frame, (new_w, new_h))

        # 모델에서 결과 얻어오기 과정
        blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)


        # 박스 추출기로 글자 위치 추정
        rectangles, confidences = box_extractor(scores, geometry, min_confidence=args['min_confidence'])

        # non-max suppression으로 박스 위치 추정 과정
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        # ROI(region of interest) 얻어오기
        roi_list = []
        for (start_x, start_y, end_x, end_y) in boxes:

            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)

            dx = int((end_x - start_x) * args['padding'])
            dy = int((end_y - start_y) * args['padding'])

            start_x = max(0, start_x - dx)
            start_y = max(0, start_y - dy)
            end_x = min(orig_w, end_x + (dx * 2))
            end_y = min(orig_h, end_y + (dy * 2))

            # 분석되어야 하는 ROI 획득
            roi = orig[start_y:end_y, start_x:end_x]
            roi_list.append((roi, (start_x, start_y, end_x, end_y)))

        # ROI내의 텍스트 분석하기
        if roi_list:
            # 병렬처리 위해 8개의 멀티프로세스 돌리기
            a_pool = multiprocessing.Pool(8)
            results = a_pool.map(process_detection, roi_list)

            a_pool.close()

            # 인식된 텍스트와 박스를 화면에 그리기
            for text, box in results:
                start_x, start_y, end_x, end_y = box
                draw.text((start_x, start_y - 20), text, font=font, fill=(0, 0, 255))
                draw.rectangle((start_x, start_y, end_x, end_y), outline=(0, 255, 0), width=2)
                orig = np.array(img)


        # fps 표시
        # 현재 초단위로 시간 가져오기
        curTime = time.time()

        sec = curTime - prevTime
        #이전 시간을 현재시간으로 다시 저장시킴
        prevTime = curTime

        # 프레임 계산
        Fps = 1/(sec)
        f = "FPS : %0.1f" % Fps

        # 프레임 화면에 표시
        cv2.putText(orig, f, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        fps.update()

        # 화면으로 보여주기
        cv2.imshow("Text Detection", orig)
        key = cv2.waitKey(1) & 0xFF

        # q 키를 누르면 꺼지도록 설정
        if key == ord('q'):
            break
    
    fps.stop()

    # 웹캠 끄기
    if not args.get('video', False):
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()
