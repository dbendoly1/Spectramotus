# Motion-triggered Media Player (minimal)

Minimal skeleton that displays a base image and plays short clips when motion is detected.

Setup (Debian/Raspbian-like):

```bash
sudo apt update
sudo apt install -y python3-pip ffmpeg v4l-utils
pip3 install -r requirements.txt
```

Run:

```bash
python main.py
```

Place your clip files under `clips/` and the base image under `assets/background.jpg` or adjust `config.yaml`.
