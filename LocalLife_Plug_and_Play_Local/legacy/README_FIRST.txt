LOCAL LIFE V14 — FULL RECREATED CODEBASE

IMPORTANT
=========
This is a reconstructed version of the V14 architecture we developed in chat.
It is not guaranteed to be byte-for-byte identical to an older original file.

V14 logic
=========
Logitech C920:
- object/foreground mask
- dominant object color
- validated count source

Intel RealSense D435/D435i:
- real hardware depth
- explicit empty-scene baseline
- depth difference + intrinsics volume integration

Windows laptop:
- Depth Anything V2 worker on Logitech RGB

Dashboard:
http://192.168.0.123:5005/

Files
=====
Pi:
- locallife_v14.py
- RUN_PI_V14.sh
- requirements_pi.txt

Windows:
- da_worker_v14.py
- RUN_DA_V14.ps1
- requirements_windows.txt

WINDOWS POWERSHELL — COPY TO PI
================================
cd C:\Users\HP\LocalLife_Final

scp "C:\Users\HP\Downloads\LocalLife_V14_FULL_RECREATED\locallife_v14.py" locallife@192.168.0.123:/home/locallife/LocalLife/

scp "C:\Users\HP\Downloads\LocalLife_V14_FULL_RECREATED\RUN_PI_V14.sh" locallife@192.168.0.123:/home/locallife/LocalLife/

ssh locallife@192.168.0.123

PI TERMINAL
===========
cd /home/locallife/LocalLife
chmod +x RUN_PI_V14.sh
./RUN_PI_V14.sh

SECOND WINDOWS POWERSHELL — DA V2
==================================
cd C:\Users\HP\LocalLife_Final
Set-ExecutionPolicy -Scope Process Bypass
.\RUN_DA_V14.ps1

CONTROLLED TEST
===============
1. Remove object from the scene.
2. Click SET EMPTY SCENE.
3. Place exactly one object inside both ROIs.
4. Step away.
5. Check Logitech contour and color.
6. Check RealSense volume.
7. Click ACCEPT / COUNT OBJECT once.

WHY COUNT IS MANUAL/VALIDATED HERE
==================================
Previous versions sometimes increased count continuously from noise/motion.
This reconstruction only increments after a valid Logitech foreground object
exists and you press ACCEPT. That makes the test reproducible for thesis data.
