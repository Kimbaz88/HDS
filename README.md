## Insight App 
### Part of my project at Insight Data Science was to deploy an app that showcased P2, my preterm labor risk prediction tool.
I chose to deploy my app using Streamlit and to host it on Heroku.

My app can be found at: https://preterm-predictor.herokuapp.com.
In the current repository I have placed all the files necessary to deploy the app, as well as a py file containing the LightGBM model from which the pickle file was created.


This is the content list for the Insight App repository:
- Requirements: file listing all the packages used in the st_runner file and needed to publish the app
- st-runner: file containing the Streamlit code for the website front-end
- pp_app: file containing the LightGBM finalized model where it is pickled into the pickle file
- lgbm.pkl: file containing the pickled LightGBM model for app use

#### For all the information on the general project and the code behind making P2, please refer to my Insight-Project repository.
