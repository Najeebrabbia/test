import pickle
from sklearn.svm import SVC

from sklearn.svm import LinearSVC
def model_training():
    with open("/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data/my_data.txt","rb") as f:
        data = pickle.load(f)
    with open("/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data/my_label.txt","rb") as f:
        labels= pickle.load(f)
    classifier= SVC(C=17.0,kernel='linear', random_state=None)
    classifier.fit(data, labels)
    with open("/home/rabianajeeb/python_projects/Datasets/SER_Data/Training_Data/my_tarinmodel.pickle","wb") as f:
     pickle.dump(classifier, f)
    print("Trainig done, and model has been saved. . .")

if __name__ == "__main__":
    print("Hello word . . ..")
    model_training()
    print("done")

