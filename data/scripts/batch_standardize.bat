@echo off
echo Creating output directories...
mkdir ..\anchor_speaker_std 2>nul
mkdir ..\other_speakers_std 2>nul

echo Processing anchor_speaker files...
for %%f in (..\anchor_speaker\*.mp3) do (
    echo Converting %%f
    python standardize.py --audio %%f --output-wav ..\anchor_speaker_std\%%~nf.wav
)

echo Processing other_speakers files...
for %%f in (..\other_speakers\*.mp3) do (
    echo Converting %%f
    python standardize.py --audio %%f --output-wav ..\other_speakers_std\%%~nf.wav
)

echo Done! All files standardized.
pause