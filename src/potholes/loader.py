import json
import os
from typing import Union

import click
from pathlib import Path
import humanfriendly
import numpy as np
import pandas as pd
import tqdm
import yaml

from potholes.tools.generator import generate_samples as generate_samples_func
from potholes.tools.gps_tools import plot_route
from potholes.tools.session import load_session, find_sessions, delete_session, get_session_stats


def json_default(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


class SessionsParam(click.ParamType):
    name = "sessions"

    def convert(self, value, param, ctx):
        value = value.lower().strip()

        if value == "all":
            return "all"

        result = set()

        for part in value.split(","):
            part = part.strip()

            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    start, end = int(a), int(b)
                except ValueError:
                    self.fail(f"Invalid session range '{part}'", param, ctx)

                if start > end:
                    self.fail(f"Invalid range '{part}' (start > end)", param, ctx)

                result.update(range(start, end + 1))
            else:
                try:
                    result.add(int(part))
                except ValueError:
                    self.fail(f"Invalid session id '{part}'", param, ctx)

        return sorted(result)


def print_sessions_header():
    click.echo(
        f"{' ':<60} {'Speed':>7}"
        "\n"
        f"{'#':<3} {'Date':<10} {'Time':<8} {'Sensor':<19} {'Frames':<8} {'PotHole':>7} {'Bump':>7} {'ManHole':>7} {'Other':>7} {'Total time'}"
    )
    click.echo("-" * 120)


def print_session_row(session: dict, index: int):
    dt = session["session_start_time"]
    stats = session["stats"]
    labels = stats['labels']
    click.echo(
        f"{index:<3} "
        f"{dt.date()} "
        f"{dt.time()} "
        f"{session['sensor_name']:<19} "
        f"{stats['frames']:>8} "
        f"{labels.get('pothole', 0):>7} "
        f"{labels.get('speed_bump', 0):>7} "
        f"{labels.get('manhole', 0):>7} "
        f"{labels.get('other', 0):>7} "
        f"{humanfriendly.format_timespan(stats['time'])} "
    )


def print_sessions_table(sessions: list[dict]):
    print_sessions_header()
    for i, s in enumerate(sessions, 1):
        print_session_row(session=s, index=i)

def _format_stat_value(value: float | None, decimals: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _print_metric_block(title: str, stats: dict, order: list[str], unit: str, decimals: int):
    if not stats:
        return

    click.echo(f"{title}:")
    for key in order:
        if key not in stats:
            continue
        value_str = _format_stat_value(stats.get(key), decimals)
        click.echo(f"  {key:<7} {value_str} {unit}")
    click.echo("")


def print_session_stats(session: dict, stats: dict):
    click.echo(f"Session: {session['sensor_name']} ({session['session_key']})")
    click.echo(f"Start:   {session['session_start_time']}")
    click.echo(f"Path:    {session['session_path']}")
    click.echo("")

    frames = stats.get("frames", 0)
    duration = stats.get("time")
    click.echo(f"Frames:  {frames}")
    if duration is not None:
        click.echo(f"Duration: {humanfriendly.format_timespan(duration)}")

    overall_rate = stats.get("frame_rate_hz", {}).get("overall")
    if overall_rate is not None:
        click.echo(f"Overall rate: {_format_stat_value(overall_rate, 2)} Hz")
    click.echo("")

    _print_metric_block(
        title="Frame rate",
        stats=stats.get("frame_rate_hz", {}),
        order=["min", "mean", "median", "max", "std", "overall"],
        unit="Hz",
        decimals=2,
    )
    _print_metric_block(
        title="Frame interval",
        stats=stats.get("frame_interval_seconds", {}),
        order=["min", "mean", "median", "max", "std"],
        unit="s",
        decimals=4,
    )

    gps_missing_pct = stats.get("gps_missing_pct")
    if gps_missing_pct is not None:
        missing_rows = int(round(frames * gps_missing_pct / 100)) if frames else 0
        click.echo(f"GPS missing: {gps_missing_pct:.2f}% ({missing_rows} rows)")

    dup_ts = stats.get("duplicate_timestamps")
    if dup_ts is not None:
        click.echo(f"Duplicate timestamps: {dup_ts}")

    dup_sensor_ts = stats.get("duplicate_sensor_timestamps")
    if dup_sensor_ts is not None:
        click.echo(f"Duplicate sensor timestamps: {dup_sensor_ts}")
    click.echo("")

    labels = stats.get("labels", {}) or {}
    click.echo("Labels:")
    if not labels:
        click.echo("  (none)")
        return

    label_order = ["pothole", "speed_bump", "manhole", "other"]
    ordered = label_order + sorted([k for k in labels.keys() if k not in label_order])
    for label in ordered:
        click.echo(f"  {label:<12} {labels.get(label, 0)}")


@click.group()
def cli():
    """
    Pothole detector data loader CLI
    """
    pass

@cli.command(help="Lists all sessions found in a given folder", name="list")
@click.argument("folder",type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
def list_sessions(folder: Path):
    sessions = find_sessions(str(folder))
    print_sessions_table(sessions)


@cli.command(help="Get several statistics for a session", name="stats")
@click.argument("folder",type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
                metavar="FOLDER")
@click.argument("session_number", type=int, required=True, metavar="SESSION_NUMBER")
def session_stats(folder: Path, session_number: int):
    sessions = find_sessions(str(folder))
    session = sessions[session_number - 1]
    stats = session.get("stats") or get_session_stats(session)
    print_session_stats(session=session, stats=stats)


@cli.command()
@click.argument("folder",type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
                metavar="FOLDER")
@click.argument("session_list", type=SessionsParam(), required=True,
                metavar="SESSION_LIST")
@click.option("-o", "--output-folder", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Output folder for the session file. Defaults to session folder.")
@click.option("-sr", "--sample-rate", type=int, default=50, help="Resample rate in Hz (default 50)")
@click.option("-g", "--generate-samples", is_flag=True, default=False,
              help="Generate sample files for the selected sessions")
@click.option("-ws", "--window-size", type=int, default=20,
              help="Window size in seconds for the sample files (default 20)")
@click.option("--step", type=int, default=1,
              help="Seconds between consecutive windows for the sample files (default 1) ")
@click.option("-v", "--verbose", is_flag=True,
              help="Prints load and parse debug information")
def process(folder: Path, session_list: Union[list[int]|str], output_folder: Path, sample_rate: int, verbose: bool,
            generate_samples: bool, window_size: int, step: int):
    """
    Preprocess a session and generate a data file with all the session data.

    FOLDER Path to a folder containing raw sensor session files.

    SESSION_LIST Which sessions to process.

    \b
        Accepted formats:
          all            Process all detected sessions
          3              Process session 3 only
          1,2,5          Process multiple sessions
          2-6            Process a range of sessions
          1,3-5,8        Mixed list and ranges
    """
    sessions = find_sessions(str(folder))

    if session_list == "all":
        session_list = list(range(len(sessions)))
    else:
        session_list = [x - 1 for x in session_list]

    for session_number in session_list:
        session = sessions[session_number]

        # prepare session folders and file names
        o_folder = session["session_path"] if output_folder is None else output_folder
        session_name = f"session_{session['sensor_name']}_{session['session_key']}".replace(" ", "")
        d_folder = os.path.join(o_folder, session_name)
        os.makedirs(d_folder, exist_ok=True)

        # load the session and generate session data
        click.echo(f"Loading {session['sensor_name']:<19} {session['session_start_time']:}")
        data = load_session(session=session, sample_rate=sample_rate, verbose=verbose)
        click.echo(f"Total frames: {len(data)}")

        # save session file
        session_file = os.path.join(d_folder, f"{session_name}.csv")
        data.to_csv(session_file, index=False)
        click.echo(f"Session saved to {session_file}", nl=True)

        # save session map
        map = plot_route(data)
        output_map = os.path.join(d_folder, "route_map.html")
        map.save(output_map)
        click.echo(f"Session route map saved to {output_map}", nl=True)

        if generate_samples:
            # generate samples data for session
            total_samples, iterator, spec_params = generate_samples_func(data, window_size=window_size, step=step)
            click.echo(f"Generating data samples for session")
            with tqdm.tqdm(total=total_samples) as pbar:
                for sample, meta in iterator:
                    labels = meta["labels"]
                    label_str = "+".join(sorted({e["label"] for e in labels})) if len(labels) > 0 else "normal"
                    start_idx = meta["start_idx"]
                    output_file = os.path.join(d_folder, f"data_{start_idx}_{window_size}_{step}_{label_str}.npz")

                    meta["sensor_name"] = session["sensor_name"]
                    meta_str = json.dumps(meta, ensure_ascii=False, default=json_default)
                    np.savez_compressed(output_file, sample=sample, meta=meta_str)

                    pbar.update(1)
                    pbar.set_description(f"Generated data_{start_idx}_{window_size}_{step}_{label_str}.npz", refresh=True)

            # save spectrogram params
            spec_params_file = os.path.join(d_folder, "spectrogram_params.yaml")
            with open(spec_params_file, "w") as f:
                yaml.dump(spec_params, f)


@cli.command(help="Deletes a session")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.argument("session_number", type=int, required=True)
def delete(folder: Path, session_number: int):
    session_number = int(session_number) - 1
    sessions = find_sessions(str(folder))
    session = sessions[session_number]

    print_sessions_header()
    print_session_row(session=session, index=session_number)
    click.echo("")

    if not click.confirm("Are you sure you want to delete this session?"):
        click.echo("Aborted", err=True, nl=True)
        return

    delete_session(session=session)
    click.echo("")
    click.echo(f"Deleted session {session['session_key']} in folder {session['session_path']}", nl=True)


if __name__ == "__main__":
    cli()
