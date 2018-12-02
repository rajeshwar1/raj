from keras.models import model_from_json
model.load_weights('vgg_face_weights.h5')
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
img1_representation = vgg_face_descriptor.predict(preprocess_image('1.jpg'))[0,:]
img2_representation = vgg_face_descriptor.predict(preprocess_image('2.jpg'))[0,:]

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
epsilon = 0.40 #cosine similarity
#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
 img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
 img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
 
 cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
 euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
 
 if(cosine_similarity < epsilon):
  print("verified... they are same person")
 else:
  print("unverified! they are not same person!")
