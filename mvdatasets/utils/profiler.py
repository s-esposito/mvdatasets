import time


class Profiler:
    """
    Profiler class to measure time of different parts of the code.
    """

    def __init__(self, verbose=False):
        self.active = {}  # store active timers start time
        self.history = {}  # keep avg time for each name
        self.verbose = verbose

    def reset(self):
        """
        Resets the profiler.
        """
        self.active = {}
        self.history = {}

    def start(self, name):
        """
        Starts a timer with the given name.
        """
        self.active[name] = time.time()

    def end(self, name):
        """
        Ends a timer with the given name, updates the history.
        """
        elapsed = time.time() - self.active[name]

        if name in self.history:
            # read previous state, update it and write it back
            state = self.history[name]
            state["count"] += 1
            state["last"] = elapsed
            state["sum"] += elapsed
            state["avg"] = state["sum"] / state["count"]
            self.history[name] = state
        else:
            # create new state
            self.history[name] = {
                "count": 1,
                "last": elapsed,
                "sum": elapsed,
                "avg": elapsed,
            }

        if self.verbose:
            print(f"{name} took {elapsed} seconds")

        self.active.pop(name)

    def get_last_time(self, name):
        """
        Returns the last time for the given timer.
        """
        last = -1.0
        if name in self.history:
            state = self.history[name]
            last = state["last"]
        return last

    def get_avg_time(self, name):
        """
        Returns the average time for the given timer.
        """
        avg = -1.0
        if name in self.history:
            state = self.history[name]
            avg = state["avg"]
        return avg

    def print_avg_times(self):
        """
        Prints the average time for each timer.
        Thread timers are averaged over all threads.
        """
        print("\nPROFILER AVG TIMES")
        processed = {}  # threads aggregated
        for name, state in self.history.items():
            # check if end of name contains a number
            if name.split("_")[-1].isdigit():  # thread id
                # remove the number from the end of the name
                key = name[: -(len(name.split("_")[-1]) + 1)]
            else:
                key = name

            # average over all threads
            if key in processed:
                # read processed state, update it and write it back
                prev_state = processed[key]
                prev_state["count"] += state["count"]
                prev_state["sum"] += state["sum"]
                prev_state["avg"] = prev_state["sum"] / prev_state["count"]
                processed[key] = state
            else:
                processed[key] = state

        for name, state in processed.items():
            print(f"{name} took {state['avg'] * 1000} ms", f"{1/state['avg']:.2f} it/s")
        print("")
