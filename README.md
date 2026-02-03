# Football Charts Generator (Streamlit) - v3

## Fix for "square pitch" PNG
Your tagging coordinates are 0–100 for both axes, which makes a *square* pitch if drawn literally.
This version adds a **Pitch shape** option:

- Rectangular (recommended): scales Y from 0–100 to 0–(pitch_width), default 64.
- Square (0–100): draws a square pitch.

### Why this works
- X stays 0–100 (so left/right is correct)
- Y is scaled to make the pitch look realistic in the exported PNG/PDF.

## Run
python -m pip install -r requirements.txt
python -m streamlit run app.py
