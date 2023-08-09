from deepface import DeepFace


result = DeepFace.verify("faces/person (30).jpg", "faces/person (28).jpg","Facenet")
print(result)