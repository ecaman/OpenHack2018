from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")
myenv.add_conda_package("keras")
myenv.add_conda_package("pandas")
myenv.add_conda_package("PIL")
myenv.add_conda_package("click")
myenv.add_conda_package("keras")
myenv.add_conda_package("numpy")
myenv.add_conda_package("cv2")
myenv.add_conda_package("pickle")
myenv.add_conda_package("glob")


with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())