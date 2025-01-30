This is Max's and my Stem Fair project for 2024-2025
dyslexia_detector is the main file where we put the website together with the eye tracking and the AI model. 
train_model.py is the file where we trained and created our model using our dataset from googlesheets - we have to run this on terminal before running our website to iniate the AI model before running the website itself.
test_run.py is the file we used to test the AI model and get the average accuracy.
The requirements.txt file is a text file for all the download requirements.
The packages.txt file is the packages needed for this project.
The path_to_ETDD70_dataset.csv file is the file that contains our data set for the AI model.
The path_to_test_data.csv file is the file that contains the data that we needed for testing and getting an average accuracy
All other folders are used for either debugging, for testing or holding data from eye tracking. 
This code only works on our computers though because we downloaded all the libraries and we had to use a local url because a public streamlit website doesn't work well with openCV which is a library we are using for the eye tracking. 
Thanks for reading this!
