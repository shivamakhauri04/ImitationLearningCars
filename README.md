# Imitation_learning_for_SelfDrvingCars


Demo steps:

Basic Requirements
1. Ubuntu
2. GPU with cuda 10.1 - The codes are gpu intensive

1. sudo apt update
2. sudo apt install python3-dev python3-pip
3. sudo pip3 install -U virtualenv
4. virtualenv --system-site-packages -p python3 ./venv
5. source ./venv/bin/activate
6. pip3 install torch torchvision (GPU with Cuda 10.1)
7. pip install opencv-contrib-python
8. pip install scipy
9. pip install pandas
10. pip install matplotlib
11. cd <repository>
12. python infer_real.py
< This runs the inference code. It will print the various steeting angles predicted by my model on the terminal for various test images and save a video at "imitation.avi" in the folder. Run it by VLC to see the results. Attaching the video too in this repo.
13. # to run training code. Download the open source training data from kaggle. It may ask you to create a account.
Here is the link to the training set https://www.kaggle.com/zaynena/selfdriving-car-simulator/download
14. Extracct and Save the folder in your repo. The folder name should "driving_dataset"
15. python train_real.py
Your training would start. You can go back to infer_real.py to test the performance of your trained model
