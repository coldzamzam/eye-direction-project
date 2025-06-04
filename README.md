**EYE DIRECTION DETECTION PROJECT**

If you just want to run the program **(1st Method)**
1. Just fork or clone this repository.
2. Go to <a href="https://share.streamlit.io/">Streamlit Website</a> and login with your GitHub.
3. Create app, select repository, and deploy app.

<hr>

If you want to train the model too **(2nd Method)**
1. Fork or clone this repository
2. Download the dataset <a href="https://drive.google.com/file/d/1zwKYWjJCLqzYg3aBpyur-EZpBGvFwxBp/view?usp=sharing">here</a> (that's contains over 10K dataset, I got it from Kaggle).
3. Copy the dataset in your subfolder project (yea i know i use the conventional way, cz why not).
   ```
   your-project/
     dataset/
       closed_look/
       forward_look/
       left_look/
       right_look/
     model/
       direction_model.pt
   ```
   direction_model.pt is a file that stores the model after we train the model.
5. Run train_model.py
   ```
   python train_model.py
   ```
   You can change the epochs cuz its way tooo loonggg to wait even its just 10 epochs.
   Ah, GitHub won't accept if we push files over 100 MB, so just upload your direction_model.pt and replace the google drive link ID in app.py with yours. (If your model is under 100 MB, just ignore it)
7. Follow step 2 and 3 in the 1st method.

*Notes:
For my friends, just clone this and chat me.
