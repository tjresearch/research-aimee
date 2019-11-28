from imageai.Prediction import ImagePrediction
import os
from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./resnet50_coco_best_v2.0.1.h5"
input_path = "./001.jpg"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])


current_dir = os.getcwd()
image_predictions = ImagePrediction()
image_predictions.setModelTypeAsResNet()
image_predictions.setModelPath(os.path.join(current_dir, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
image_predictions.loadModel()
images_array = list(filter(lambda x: x.endswith(".jpg") or x.endswith(".png"),os.listdir(current_dir)))
results_array = image_predictions.predictMultipleImages(images_array, result_count_per_image=10)
def consolidate_prediction_results(results):
    list_of_prediction = []
    for each_result in results:
        predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
        list_of_prediction.append(predictions)
        for index in range(len(predictions)):
            print(predictions[index] , " : " , percentage_probabilities[index])
        print("______________________________")
    print('set_of_prediction:',set([item for items in list_of_prediction for item in items]))

print(results_array)
consolidate_prediction_results(results_array)