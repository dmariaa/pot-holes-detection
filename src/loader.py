import os
from typing import Union

import click
from pathlib import Path
import humanfriendly
import numpy as np
import tqdm
import yaml

from tools.generator import generate_samples
from tools.gps_tools import plot_route
from tools.session import load_session, find_sessions, delete_session


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


@cli.command()
@click.argument("folder",type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
                metavar="FOLDER")
@click.argument("session_list", type=SessionsParam(), required=True,
                metavar="SESSION_LIST")
@click.option("-o", "--output-folder", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Output folder for the session file. Defaults to session folder.")
@click.option("-sr", "--sample-rate", type=int, default=50, help="Resample rate in Hz (default 50)")
@click.option("-ws", "--window-size", type=int, default=20, help="Window size in seconds (default 20)")
@click.option("--step", type=int, default=1, help="Seconds between consecutive windows (default 1) ")
@click.option("-v", "--verbose", is_flag=True, help="Prints load and parse debug information")
def process(folder: Path, session_list: Union[list[int]|str], output_folder: Path, sample_rate: int, verbose: bool,
            window_size: int, step: int):
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
        session_file = os.path.join(o_folder, f"{session_name}.csv")
        data.to_csv(session_file, index=False)
        click.echo(f"Session saved to {session_file}", nl=True)

        # save session map
        map = plot_route(data)
        output_map = os.path.join(d_folder, "route_map.html")
        map.save(output_map)
        click.echo(f"Session route map saved to {output_map}", nl=True)

        # generate samples data for session
        total_samples, iterator, spec_params = generate_samples(data, window_size=window_size, step=step)
        click.echo(f"Generating data samples for session")
        with tqdm.tqdm(total=total_samples) as pbar:
            for sample, label, start_idx in iterator:
                output_file = os.path.join(d_folder, f"data_{start_idx}_{window_size}_{step}_{label}.npy")
                np.save(output_file, sample)
                pbar.update(1)
                pbar.set_description(f"Generated data_{start_idx}_{window_size}_{step}_{label}.npy", refresh=True)

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
