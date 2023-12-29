###############################################################################################################
# 라이브러리
import cv2
import math
import pyttsx3
import threading
from pathlib import Path
import pygame
import sys
import time
import os
###############################################################################################################
# 전역변수 설정
player_neck = 0
global_frame_width = 1280
global_frame_height = 960
pose_detected = False
###############################################################################################################
# 각 파트 번호 BODY_PARTS, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
###############################################################################################################
# 각 파일 path
BASE_DIR = Path(__file__).resolve().parent
protoFile = str(BASE_DIR)+"/models/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = str(BASE_DIR)+"/models/pose_iter_160000.caffemodel"

# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#벡엔드로 쿠다를 사용하여 속도향상
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# 쿠다 디바이스에 계산 요청
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
###############################################################################################################
# 카메라 속성 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
inputWidth=320;
inputHeight=240;
inputScale=1.0/255;
###############################################################################################################
# 각도 추출 함수
def calculate_angle(point1, point2, point3):
    # 벡터 A와 B 생성
    vectorA = [point2[0] - point1[0], point2[1] - point1[1]]
    vectorB = [point3[0] - point2[0], point3[1] - point2[1]]
    # 벡터의 내적
    dot_product = vectorA[0]*vectorB[0] + vectorA[1]*vectorB[1]
    # 벡터의 크기(길이)
    magnitudeA = math.sqrt(vectorA[0]**2 + vectorA[1]**2)
    magnitudeB = math.sqrt(vectorB[0]**2 + vectorB[1]**2)
    # 두 벡터 사이의 각도 계산
    angle = math.acos(dot_product / (magnitudeA * magnitudeB))
    # 라디안을 도로 변환
    angle = math.degrees(angle)
    return angle
###############################################################################################################
# pyttsx3 엔진 초기화
engine = pyttsx3.init()
lock = threading.Lock()

# 별도의 스레드에서 음성 출력을 처리하는 함수
def speak_async(text):
    with lock:
        engine.say(text)
        engine.runAndWait()
        
# 음성 출력 함수 수정
def speak(text):
    # 스레드 생성 및 시작
    t = threading.Thread(target=speak_async, args=(text,))
    t.start()
###############################################################################################################
# 관절 검출 함수
def run_pose():
    global player_neck, global_frame_height, global_frame_width
    global pose_detected
    # 창 이름 및 초기 위치 설정
    window_name = "Output - KeyPoints"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 80)
    #반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
    while cv2.waitKey(1) <0:  #아무 키나 누르면 끝난다.
        #웹캠으로부터 영상 가져옴
        hasFrame, frame = capture.read()
        
        #웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
        if not hasFrame:
            cv2.waitKey()
            break
        
        # 카메라에서 프레임 크기 가져오기
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        global_frame_width = frameWidth
        global_frame_height = frameHeight
        
        # 이미지 전처리
        inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    
        # 처리된 이미지 inpBlob을 신경망의 입력으로 설정하여 network에 넣어주기
        net.setInput(inpBlob)
        
        # 결과 반환
        output = net.forward()

        # 키포인트 검출시 이미지에 그려줌
        points = []
        for i in range(0,15):
            # 해당 신체부위 키포인트 검출
            probMap = output[0, i, :, :]

            # 키포인트 위치와 신뢰도 확인
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # 원래 이미지에 맞게 점 위치 조정
            x = (frameWidth * point[0]) / output.shape[3]
            y = (frameHeight * point[1]) / output.shape[2]

            # 신뢰도가 0.1보다 크면 point에 추가, 검출했는데 부위가 없으면 None
            if prob > 0.1 :
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
    
        # 각 POSE_PAIRS별로 선 그어줌
        for pair in POSE_PAIRS:
            partA = pair[0]             # Head
            partA = BODY_PARTS[partA]   # 0
            partB = pair[1]             # Neck
            partB = BODY_PARTS[partB]   # 1
        
            #partA와 partB 사이에 선을 그어줌
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

        # BODY_PARTS에서 각 관절의 인덱스를 찾음
        Head_index = BODY_PARTS["Head"]
        Neck_index = BODY_PARTS["Neck"]
        RShoulder_index = BODY_PARTS["RShoulder"]
        RElbow_index = BODY_PARTS["RElbow"]
        RWrist_index = BODY_PARTS["RWrist"]
        LShoulder_index = BODY_PARTS["LShoulder"]
        LElbow_index = BODY_PARTS["LElbow"]
        LWrist_index = BODY_PARTS["LWrist"]
        RHip_index = BODY_PARTS["RHip"]
        RKnee_index = BODY_PARTS["RKnee"]
        RAnkle_index = BODY_PARTS["RAnkle"]
        LHip_index = BODY_PARTS["LHip"]
        LKnee_index = BODY_PARTS["LKnee"]
        LAnkle_index = BODY_PARTS["LAnkle"]
        Chest_index = BODY_PARTS["Chest"]

        # points 리스트에서 해당 관절의 좌표를 가져옴
        Head = points[Head_index]
        Neck = points[Neck_index]
        RShoulder = points[RShoulder_index]
        RElbow = points[RElbow_index]
        RWrist = points[RWrist_index]
        LShoulder = points[LShoulder_index]
        LElbow = points[LElbow_index]
        LWrist = points[LWrist_index]
        RHip = points[RHip_index]
        RKnee = points[RKnee_index]
        RAnkle = points[RAnkle_index]
        LHip = points[LHip_index]
        LKnee = points[LKnee_index]
        LAnkle = points[LAnkle_index]
        Chest = points[Chest_index]
        
        # Pygame 시작을 위한 검출 여부
        if Head is not None and Neck is not None and Chest is not None and RKnee is not None:
            pose_detected = True
        
        # Pygame을 위한 Neck의 좌표 업데이트
        if Neck is not None:
            player_neck = Neck[1]
        
        # 음성 출력
        if Chest is not None and RHip is not None and RKnee is not None:
            back_angle = calculate_angle(Chest, RHip, RKnee)
            if 45 < back_angle < 90:
                speak("허리를 일자로 펴세요")
                print('각도:' , back_angle)
        
        # 출력창 띄우기
        cv2.imshow(window_name,frame)

    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    # 모든 윈도우 창 닫음
    cv2.destroyAllWindows()
###############################################################################################################
# 게임 실행 함수
def run_game():
    global player_y
    global player_neck
    global global_frame_height
    global last_obstacle_time
    global score
    global obstacle_state
    global pose_detected
    
    # Pygame 창의 초기 위치 설정
    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (1290, 110)
    
    # Pygame 초기화
    pygame.init()

    # 화면 설정
    WIDTH, HEIGHT = 410, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Avoid the Obstacles")

    # 색깔 정의
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    
    # 플레이어 설정 (원으로 변경)
    player_radius = 25
    player_x = WIDTH // 2
    player_y = HEIGHT // 2
    player_speed = 5

    # 장애물 설정
    obstacle_width, obstacle_height = 20, 360  # 장애물의 높이를 250으로 조정
    obstacle_speed = 3
    obstacle_frequency = 3000  # 밀리세컨 단위 주기로 장애물 생성
    obstacles = []

    # 점수
    score = 0
    font = pygame.font.SysFont(None, 30)

    # 장애물 상태
    obstacle_state = "up"

    # 마지막으로 장애물이 생성된 시간 기록
    last_obstacle_time = pygame.time.get_ticks()

    # 게임 루프
    clock = pygame.time.Clock()
    
    # 검출 못하면 대기
    while not pose_detected:
        time.sleep(0.1)
    
    # 검출되면 시작
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 사용자의 목과 점 동기화
        player_y = HEIGHT * (player_neck / global_frame_height)
        
        # 플레이어 화면 밖으로 나가지 않도록 제한
        player_y = max(player_radius, min(HEIGHT - player_radius, player_y))

        # 일정한 주기로 장애물 생성
        current_time = pygame.time.get_ticks()
        if current_time - last_obstacle_time >= obstacle_frequency:
            obstacle_x = WIDTH
            obstacle_y = HEIGHT - obstacle_height if obstacle_state == "up" else 0
            obstacles.append((obstacle_x, obstacle_y))
            last_obstacle_time = current_time
            # 상태 변경
            obstacle_state = "up" if obstacle_state == "down" else "down"

        # 충돌 여부를 나타내는 변수 추가
        collision = False  

        # 장애물 이동 및 충돌 체크
        for i in range(len(obstacles)):
            obstacles[i] = (obstacles[i][0] - obstacle_speed, obstacles[i][1])
            
            # 원과 직사각형의 충돌 감지
            obstacle_left = obstacles[i][0]
            obstacle_right = obstacles[i][0] + obstacle_width
            obstacle_top = obstacles[i][1]
            obstacle_bottom = obstacles[i][1] + obstacle_height
            player_left = player_x - player_radius
            player_right = player_x + player_radius
            player_top = player_y - player_radius
            player_bottom = player_y + player_radius

            if not (
            player_right < obstacle_left
            or player_left > obstacle_right
            or player_bottom < obstacle_top
            or player_top > obstacle_bottom
            ):
                collision = True  # 충돌이 감지되면 collision 변수를 True로 설정
            
            if obstacles[i][0] + obstacle_width < 0:
                obstacles.pop(i)
                score += 1
                break

        # 그리기
        screen.fill(WHITE)
        pygame.draw.circle(screen, BLUE, (player_x, player_y), player_radius)

        for obstacle in obstacles:
            pygame.draw.rect(
                screen, RED, (obstacle[0], obstacle[1], obstacle_width, obstacle_height)
            )

        score_display = font.render(f"Score: {score}", True, RED)
        screen.blit(score_display, (10, 10))
        
        # 충돌이 감지되면 게임 종료
        if collision:
            print("Game Over! Your final score:", score)
            pygame.quit()
            sys.exit()
         
        pygame.display.flip()
        clock.tick(30)
###############################################################################################################
# 메인
if __name__ == '__main__':
    # 스레드 생성
    pose_thread = threading.Thread(target=run_pose)
    
    # 스레드 시작
    pose_thread.start()
    run_game()
    pose_thread.join()
###############################################################################################################