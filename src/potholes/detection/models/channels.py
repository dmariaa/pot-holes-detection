SIGNAL_CHANNELS = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]


def resolve_channel_indices(channels: str | int | list[str | int] | None) -> tuple[list[int], list[str]]:
    if isinstance(channels, (str, int)):
        channels = [channels]

    if channels is None or len(channels) == 0 or channels == ["all"]:
        indices = list(range(len(SIGNAL_CHANNELS)))
        return indices, SIGNAL_CHANNELS.copy()

    indices = []
    names = []
    for channel in channels:
        if isinstance(channel, int):
            idx = channel
        elif isinstance(channel, str):
            if channel.isdigit():
                idx = int(channel)
            else:
                if channel not in SIGNAL_CHANNELS:
                    raise ValueError(f"Unknown channel '{channel}'. Valid channels: {SIGNAL_CHANNELS}")
                idx = SIGNAL_CHANNELS.index(channel)
        else:
            raise TypeError(f"Channel must be a string name or integer index, got {type(channel).__name__}")

        if idx < 0 or idx >= len(SIGNAL_CHANNELS):
            raise ValueError(f"Channel index {idx} is out of range 0-{len(SIGNAL_CHANNELS) - 1}")
        if idx in indices:
            continue

        indices.append(idx)
        names.append(SIGNAL_CHANNELS[idx])

    if len(indices) == 0:
        raise ValueError("At least one input channel must be selected")

    return indices, names
