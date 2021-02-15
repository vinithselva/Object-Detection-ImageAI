from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./models/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
input_path = "./input/test.jpg"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])