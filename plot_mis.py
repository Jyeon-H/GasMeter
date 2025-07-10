import numpy as np
import matplotlib.pyplot as plt

def plot_misclassified(model, x_test, y_test):
    prediction = model.predict(x_test)
    predict_test = np.argmax(prediction, axis=-1)

    y = np.array(np.where(y_test == 1))

    predict_right = np.equal(predict_test, y[1])
    
    predict_wrong = np.array(np.where(predict_right == False))
    predict_wrong = predict_wrong.reshape(-1)

    plt.figure(figsize=(15, 5))
    for i in range(12):
        plt.subplot(2, 6, i+1)
        predict_wrongs = predict_wrong[i]
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[predict_wrongs], cmap='gray') 
        plt.title('Predicted: {} / Label: {}'.format(predict_test[predict_wrongs], y[1][predict_wrongs]))
    plt.show()