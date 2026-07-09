from pathlib import Path
import re
import shutil

import click

from potholes.tools.session import find_sessions, format_duration_hms, format_number, load_session


def markdown_cell(value) -> str:
    return str(value).replace("|", "\\|")


def markdown_link_cell(value, link: str) -> str:
    return f"[{markdown_cell(value)}]({link})"


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def session_page_slug(session: dict) -> str:
    folder_name = Path(session["session_path"]).name
    return slugify(f"{folder_name}_{session['session_key']}")


def session_row(session: dict, index: int) -> list[str]:
    dt = session["session_start_time"]
    stats = session["stats"]
    labels = stats["labels"]
    folder_name = Path(session["session_path"]).name

    return [
        str(index),
        folder_name,
        str(dt.date()),
        str(dt.time()),
        session["sensor_name"],
        format_number(stats["frames"]),
        format_number(labels.get("pothole", 0)),
        format_number(labels.get("speed_bump", 0)),
        format_number(labels.get("manhole", 0)),
        format_number(labels.get("other", 0)),
        format_duration_hms(stats["time"]),
    ]


def video_matches_session(video_file: Path, session: dict) -> bool:
    slug = session_page_slug(session).lower()
    session_key = session["session_key"].lower()
    name = video_file.stem.lower()
    return slug in name or session_key in name


def find_session_video(doc_output_folder: Path, session: dict) -> Path | None:
    for video_file in sorted(doc_output_folder.rglob("*.mp4")):
        if video_matches_session(video_file, session):
            return video_file

    return None


def copy_session_video(video_folder: Path | None, doc_output_folder: Path, session: dict) -> Path | None:
    if video_folder is None:
        return find_session_video(doc_output_folder=doc_output_folder, session=session)

    for video_file in sorted(video_folder.rglob("*.mp4")):
        if not video_matches_session(video_file, session):
            continue

        videos_output_folder = doc_output_folder / "videos"
        videos_output_folder.mkdir(parents=True, exist_ok=True)
        output_file = videos_output_folder / f"{session_page_slug(session)}.mp4"
        shutil.copy2(video_file, output_file)
        return output_file

    return find_session_video(doc_output_folder=doc_output_folder, session=session)


def generate_session_video_file(
    session: dict,
    doc_output_folder: Path,
    sample_rate: int,
    video_window_size: int,
    overwrite_videos: bool,
) -> Path:
    from potholes.tools.video import generate_session_video

    videos_output_folder = doc_output_folder / "videos"
    videos_output_folder.mkdir(parents=True, exist_ok=True)

    output_file = videos_output_folder / f"{session_page_slug(session)}.mp4"
    if output_file.exists() and not overwrite_videos:
        return output_file

    click.echo(f"Generating video for {session['session_key']} -> {output_file}")
    session_data = load_session(session=session, sample_rate=sample_rate, verbose=False)
    title = f"{session['sensor_name']} {session['session_key']}"
    generate_session_video(
        session_data=session_data,
        out_mp4=str(output_file),
        window_size=video_window_size,
        title_prefix=title,
    )
    return output_file


def resolve_session_video(
    session: dict,
    doc_output_folder: Path,
    video_folder: Path | None,
    generate_videos: bool,
    sample_rate: int,
    video_window_size: int,
    overwrite_videos: bool,
) -> Path | None:
    video_file = copy_session_video(
        video_folder=video_folder,
        doc_output_folder=doc_output_folder,
        session=session,
    )
    if video_file is not None:
        return video_file

    if not generate_videos:
        return None

    return generate_session_video_file(
        session=session,
        doc_output_folder=doc_output_folder,
        sample_rate=sample_rate,
        video_window_size=video_window_size,
        overwrite_videos=overwrite_videos,
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")

    return lines


def build_sessions_page(session_folder: Path, sessions: list[dict]) -> str:
    headers = [
        "#",
        "Folder",
        "Date",
        "Time",
        "Sensor",
        "Frames",
        "PotHole",
        "Bump",
        "ManHole",
        "Other",
        "Total time",
    ]

    lines = [
        "# Sessions",
        "",
        f"Source folder: `{session_folder}`",
        "",
    ]

    rows = []
    for index, session in enumerate(sessions, 1):
        link = f"{session_page_slug(session)}.md"
        rows.append([markdown_link_cell(value, link) for value in session_row(session, index)])

    lines.extend(markdown_table(headers, rows))

    return "\n".join(lines) + "\n"


def metric_rows(stats: dict, keys: list[str], unit: str, decimals: int) -> list[list[str]]:
    rows = []
    for key in keys:
        if key not in stats:
            continue
        value = stats.get(key)
        value_text = "n/a" if value is None else f"{value:.{decimals}f}"
        rows.append([key, value_text, unit])
    return rows


def build_session_page(session: dict, video_file: Path | None, doc_output_folder: Path) -> str:
    dt = session["session_start_time"]
    stats = session["stats"]
    labels = stats["labels"]
    folder_name = Path(session["session_path"]).name

    summary_rows = [
        ["Folder", folder_name],
        ["Session key", session["session_key"]],
        ["Start", str(dt)],
        ["Sensor", session["sensor_name"]],
        ["Frames", format_number(stats["frames"])],
        ["Total time", format_duration_hms(stats["time"])],
        ["GPS missing", f"{stats['gps_missing_pct']:.2f}%"],
        ["Duplicate timestamps", format_number(stats["duplicate_timestamps"])],
        ["Duplicate sensor timestamps", format_number(stats["duplicate_sensor_timestamps"])],
    ]

    label_rows = [
        ["pothole", format_number(labels.get("pothole", 0))],
        ["speed_bump", format_number(labels.get("speed_bump", 0))],
        ["manhole", format_number(labels.get("manhole", 0))],
        ["other", format_number(labels.get("other", 0))],
    ]

    lines = [
        f"# {folder_name} - {session['session_key']}",
        "",
        "## Summary",
        "",
        *markdown_table(["Field", "Value"], summary_rows),
        "",
        "## Labels",
        "",
        *markdown_table(["Label", "Count"], label_rows),
        "",
        "## Frame Rate",
        "",
        *markdown_table(
            ["Metric", "Value", "Unit"],
            metric_rows(
                stats.get("frame_rate_hz", {}),
                ["min", "mean", "median", "max", "std", "overall"],
                "Hz",
                2,
            ),
        ),
        "",
        "## Frame Interval",
        "",
        *markdown_table(
            ["Metric", "Value", "Unit"],
            metric_rows(
                stats.get("frame_interval_seconds", {}),
                ["min", "mean", "median", "max", "std"],
                "s",
                4,
            ),
        ),
        "",
        "## Generated Video",
        "",
    ]

    if video_file is None:
        lines.append("No generated video found for this session.")
    else:
        video_path = "../" + video_file.relative_to(doc_output_folder).as_posix()
        lines.extend([
            f'<video controls width="100%">',
            f'  <source src="{video_path}" type="video/mp4" />',
            "</video>",
        ])

    return "\n".join(lines) + "\n"


def generate_doc(
    session_folder: Path,
    doc_output_folder: Path,
    video_folder: Path | None = None,
    generate_videos: bool = False,
    sample_rate: int = 50,
    video_window_size: int = 20,
    overwrite_videos: bool = False,
) -> Path:
    sessions = find_sessions(str(session_folder))
    doc_output_folder.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        output_file = doc_output_folder / f"{session_page_slug(session)}.md"
        video_file = resolve_session_video(
            session=session,
            video_folder=video_folder,
            doc_output_folder=doc_output_folder,
            generate_videos=generate_videos,
            sample_rate=sample_rate,
            video_window_size=video_window_size,
            overwrite_videos=overwrite_videos,
        )
        output_file.write_text(
            build_session_page(
                session=session,
                video_file=video_file,
                doc_output_folder=doc_output_folder,
            ),
            encoding="utf-8",
        )

    output_file = doc_output_folder / "index.md"
    output_file.write_text(
        build_sessions_page(session_folder=session_folder, sessions=sessions),
        encoding="utf-8",
    )

    return output_file


@click.command()
@click.argument(
    "session_folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "doc_output_folder",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--video-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder containing generated MP4 videos to copy into the docs when names match a session key.",
)
@click.option(
    "--generate-videos",
    is_flag=True,
    default=False,
    help="Generate missing MP4 videos into DOC_OUTPUT_FOLDER/videos.",
)
@click.option(
    "--sample-rate",
    type=int,
    default=50,
    show_default=True,
    help="Sample rate used when loading raw sessions for video generation.",
)
@click.option(
    "--video-window-size",
    type=int,
    default=20,
    show_default=True,
    help="Window size in seconds used for generated videos.",
)
@click.option(
    "--overwrite-videos",
    is_flag=True,
    default=False,
    help="Regenerate videos even when the target MP4 already exists.",
)
def cli(
    session_folder: Path,
    doc_output_folder: Path,
    video_folder: Path | None,
    generate_videos: bool,
    sample_rate: int,
    video_window_size: int,
    overwrite_videos: bool,
):
    """
    Generate a MkDocs markdown page listing sessions in a folder.

    SESSION_FOLDER is the raw session folder, or a parent folder containing raw
    session folders. DOC_OUTPUT_FOLDER is where index.md will be written.
    """
    output_file = generate_doc(
        session_folder=session_folder,
        doc_output_folder=doc_output_folder,
        video_folder=video_folder,
        generate_videos=generate_videos,
        sample_rate=sample_rate,
        video_window_size=video_window_size,
        overwrite_videos=overwrite_videos,
    )
    click.echo(f"Documentation written to {output_file}")


if __name__ == "__main__":
    cli()
