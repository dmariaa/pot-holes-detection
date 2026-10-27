"""Include available session videos at build time without changing Markdown sources."""
from html import escape
from pathlib import Path, PurePosixPath
import re

from mkdocs.utils import get_relative_url


def on_page_markdown(markdown, *, page, config, files):
    session_id = page.meta.get("session_id")
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        return markdown
    parent = PurePosixPath(page.file.src_uri).parent
    if len(parent.parts) != 2 or parent.parts[0] != "sessions":
        return markdown
    video_uri = (parent / "videos" / f"{session_id}.mp4").as_posix()
    video_path = Path(config["docs_dir"]) / video_uri
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return markdown
    video = files.get_file_from_path(video_uri)
    if video is None or not video.inclusion.is_included():
        return markdown
    url = escape(get_relative_url(video.url, page.url), quote=True)
    return markdown.rstrip() + (
        '\n\n## Generated Video\n\n<video controls width="100%">\n'
        f'  <source src="{url}" type="video/mp4" />\n'
        '</video>\n'
    )
