This our submission to TSYP12's EMBS challenge on mental health.

We use flutter for our app, you can just download flutter's latest version, and run the following commands (inside the adhd-diagnosis folder)
```
flutter pub get
flutter run
```
You can choose to run on a web browser, or use android emulator (we recommend android emulator, because that's the platform we developed the app on).

We use flask for our backend, we use Python 3.9.7, we recommend downloading pyenv to easily change the version (in the backend folder)

```
pyenv local 3.9.7
pip install -r requiements.txt
python app.py
```

The backend should be hosted locally: 
http://127.0.0.1:5000

Our computer vision model is in a docker container, you'd have to download it (it's 10gb :D), download Docker desktop and run this command
```
docker run --gpus all -p 9001:9001 roboflow/roboflow-inference-server-gpu
```

If you find any issues, contact: starmaq101@gmail.com
