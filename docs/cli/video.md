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
![Sample plot](../assets/cli/test_video.png)

### Background maps

Video generation downloads OpenStreetMap tiles for the route extent. Requests identify
this application as `PotholesDetection/0.1`. Tiles are cached across runs in
`%LOCALAPPDATA%/PotholesDetection/map-tiles` on Windows (or
`~/.cache/PotholesDetection/map-tiles` when `LOCALAPPDATA` is unavailable).
Keep this cache so repeated renders reuse downloaded tiles.

You can include a real contact URL or email in the application identifier by setting
`POTHOLES_MAP_USER_AGENT` before rendering. Do not impersonate a browser or another
application. See the [OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

If a video contains an “Access blocked” tile, regenerate it after correcting the map
request configuration. Existing MP4 files cannot update their background automatically.
An existing server or network block may still require resolution with the provider;
do not repeatedly retry blocked requests.

### Session videos in the documentation

Session pages store a stable `session_id`; their Markdown does not depend on local
video availability. `potholes.tools.video` writes only the requested video.
`potholes.gen_doc --generate-videos` can generate missing videos while regenerating
the session documentation, but does not embed players in the Markdown.

Place each session MP4 at `docs/sessions/<collection>/videos/<session_id>.mp4`.
During `uv run mkdocs build`, the session video hook adds a player only when that
exact file exists, is nonempty, and is included in the build. This checks availability,
not MP4 integrity. MkDocs copies the videos into the generated site.
Rebuild after adding or removing videos and deploy the entire generated site,
including its videos, together. The session videos remain ignored by Git.

Sessions with fewer than 64 processed samples (the spectrogram segment length)
are skipped with a warning; no MP4 is created and batch generation continues.
