from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task,
    filesize
)
from rich.text import Text 


def print_error(message):
    print(f"[bold red]ERROR[/bold red] {message}")
    exit(1)


def print_warning(message):
    print(f"[bold yellow]WARNING[/bold yellow] {message}")


def print_info(message):
    print(f"[bold blue]INFO[/bold blue] {message}")

    
def print_log(message):
    print(f"[bold purple]LOG[/bold purple] {message}")


def print_success(message):
    print(f"[bold green]SUCCESS[/bold green] {message}")


class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")


def progress_bar(name):
    
    # Define custom progress bar
    progress_bar = Progress(
        # name of the task
        TextColumn(f"[bold blue]{name}"),
        # progress percentage
        TextColumn("•"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        # progress bar
        BarColumn(),
        # completed iterations
        MofNCompleteColumn(),
        # elapsed time
        TextColumn("•"),
        TimeElapsedColumn(),
        # remaining time
        TextColumn("•"),
        TimeRemainingColumn(),
        # show speed
        TextColumn("•"),
        RateColumn(),
    )
    
    return progress_bar