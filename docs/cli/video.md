# Video CLI

The video CLI lives in `potholes.tools.video`. Run commands from the repository root.

```powershell
python -m potholes.tools.video --help
```

## Commands

### Generate a session video

Creates an MP4 video with the spectrogram panels and GPS route map.
Requires `ffmpeg` on PATH and downloads map tiles unless cached.

Usage:
```shell-session
$ uv run -m potholes.tools.video --help

Usage: python -m potholes.tools.video [OPTIONS] SESSION_FILE

Options:
  -o, --output PATH  Write the video to this file (extension not needed, .mp4 will be added).
  -d, --debug  Debug mode, renders only a single frame as a png.
  --help  Show this message and exit.
```

Example:
```shell-session
$ uv run -m potholes.tools.video output/session_*/session_*.csv -o out/session_video

Rendering video: 100%|##########| 240/240 [00:32<00:00,  7.45it/s]
```
<video controls width="900">
    <source src="../../assets/cli/test_video.mp4" type="video/mp4" />
</video>

### Generate a debug frame

Renders a single PNG frame instead of a full video.

Example:
```shell-session
$ uv run -m potholes.tools.video output/session_*/session_*.csv -o out/frame -d

(no stdout; writes out/frame.png)
```
![Sample plot](../../assets/cli/test_video.png)
