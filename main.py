from ObjectDetection.yolo import train_yolov5
from Classification.cnn import train_cnn
from Classification.mlp import train_mlp
from plot_mis import plot_misclassified

def main():
    print("[1단계] YOLOv5로 계량기 탐지 학습")
    train_yolov5()
    
    print("[2단계] MLP 모델로 숫자 인식 학습")
    mlp, x_test_mlp, y_test_mlp = train_mlp()
    plot_misclassified(mlp, x_test_mlp, y_test_mlp)

    print("[3단계] CNN 모델로 숫자 인식 학습")
    cnn, x_test_cnn, y_test_cnn = train_cnn()
    plot_misclassified(cnn, x_test_cnn, y_test_cnn)
    
if __name == '__main__':
    main()
