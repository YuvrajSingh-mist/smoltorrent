"""Terminal UI components."""

import io
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, RichLog, Static

from ._utils import get_logger

log = get_logger("tui")


class LogCapture(io.TextIOBase):
    """Thread-safe in-memory ring buffer for log lines.

    Implements the :class:`io.TextIOBase` interface so it can be used as
    a drop-in replacement for ``sys.stderr``.  Keeps the most recent
    *max_lines* strings; newer lines evict the oldest when the limit is
    reached.  :meth:`drain_new` returns lines accumulated since the last
    call.

    Args:
        max_lines: Maximum number of lines to retain (default 200).
    """

    def __init__(self, max_lines: int = 200) -> None:
        """Initialise an empty ring buffer.

        Args:
            max_lines: Maximum number of log lines to retain before evicting
                       the oldest (default 200).

        Returns:
            None.
        """
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.max_lines = max_lines
        self.new_lines: list[str] = []

    def write(self, s: str) -> int:
        """Append non-empty lines from *s* to the ring buffer.

        Args:
            s: String to write (may contain multiple lines or be empty).

        Returns:
            Number of characters in *s* (matches the :class:`io.TextIOBase` contract).
        """
        if not s or s == "\n":
            return len(s) if s else 0
        with self.lock:
            for line in s.splitlines():
                if line.strip():
                    self.lines.append(line)
                    self.new_lines.append(line)
                    if len(self.lines) > self.max_lines:
                        self.lines.pop(0)
        return len(s)

    def flush(self) -> None:
        """No-op flush to satisfy the :class:`io.TextIOBase` interface.

        Args:
            None.

        Returns:
            None.
        """
        pass

    def drain_new(self) -> list[str]:
        """Return and clear lines accumulated since the last call.

        Returns:
            List of new log lines (may be empty).
        """
        with self.lock:
            lines = list(self.new_lines)
            self.new_lines.clear()
            return lines


class ClusterRow:
    """Lightweight wrapper that adds computed ``age`` and ``full``
    properties to a raw cluster dict from :class:`MasterBrowser`."""

    def __init__(self, data: dict) -> None:
        """Wrap a raw cluster dict from :class:`MasterBrowser`.

        Args:
            data: Cluster dict with keys ``name``, ``uid``, ``current``,
                  ``hostname``, and ``started``.

        Returns:
            None.
        """
        self.name = data.get("name", "?")
        self.uid = data.get("uid", "?")
        self.current = int(data.get("current", 0))
        # self.expected = int(data.get("expected", 0))
        self.hostname = data.get("hostname", "?")
        self.started = float(data.get("started", time.time()))
        self.data = data

    @property
    def age(self) -> str:
        """Human-readable elapsed time since the cluster was first seen."""
        ago = time.time() - self.started
        if ago < 60:
            return f"{int(ago)}s"
        elif ago < 3600:
            return f"{int(ago // 60)}m"
        return f"{int(ago // 3600)}h"

    @property
    def full(self) -> bool:
        """``True`` when all expected workers have joined."""
        # return self.current >= self.expected
        return False


class JoinApp(App):
    """Textual TUI for selecting a smoltorrent master to join.

    Displays a table of masters discovered via mDNS.  Arrow keys navigate,
    Enter selects, ``q`` quits without selecting.

    Args:
        browser: A :class:`~discovery.grove._mdns.MasterBrowser` instance
                 that provides :meth:`get_clusters`.
    """
    COMMANDS = set()
    CSS = """
    Screen { background: $surface; }
    #title { padding: 1 2; }
    #hint { color: $text-muted; padding: 0 2 1 2; }
    DataTable { height: 1fr; }
    DataTable > .datatable--cursor { background: $accent; color: $text; }
    """

    BINDINGS = [Binding("q", "quit", "Quit", show=False)]

    def __init__(self, browser: Any) -> None:
        """Initialise the JoinApp with a live master browser.

        Args:
            browser: A :class:`~discovery.grove._mdns.MasterBrowser` that
                     provides a ``get_clusters()`` method.

        Returns:
            None.
        """
        super().__init__()
        self.browser = browser
        self.selected_cluster: Optional[dict] = None
        self.clusters: list[dict] = []
        log.info("[tui] JoinApp initialised")

    def compose(self) -> ComposeResult:
        """Build the widget tree for the JoinApp screen.

        Args:
            None.

        Returns:
            :class:`~textual.app.ComposeResult` yielding the title, cluster
            table, and hint bar widgets.
        """
        yield Static("[b]grove[/]  select a cluster", id="title")
        table = DataTable(id="clusters")
        table.cursor_type = "row"
        table.add_columns("Name", "ID", "Nodes", "Host", "Age")
        yield table
        yield Static("  [dim]↑↓ select   enter join   q quit[/]", id="hint")

    def on_mount(self) -> None:
        """Set up a 1-second refresh interval and focus the cluster table on mount.

        Args:
            None.

        Returns:
            None.
        """
        self.set_interval(1.0, self.refresh_table)
        self.refresh_table()
        self.query_one("#clusters", DataTable).focus()

    def refresh_table(self) -> None:
        """Repopulate the cluster table from the latest browser snapshot.

        Args:
            None.

        Returns:
            None.
        """
        self.clusters = self.browser.get_clusters()
        table = self.query_one("#clusters", DataTable)
        cursor = table.cursor_row
        table.clear()
        for c in self.clusters:
            row = ClusterRow(c)
            style = "dim" if row.full else ""
            nodes_style = "dim" if row.full else "green"
            table.add_row(
                Text(row.name, style=f"bold {style}" if not row.full else style),
                Text(row.uid, style="cyan dim"),
                Text(f"{row.current}", style=nodes_style),
                Text(row.hostname, style=style),
                Text(row.age, style="dim"),
            )
        if cursor is not None and self.clusters:
            table.move_cursor(row=min(cursor, len(self.clusters) - 1))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Store the selected cluster and exit the app when a row is confirmed.

        Args:
            event: Textual :class:`~textual.widgets.DataTable.RowSelected` event.

        Returns:
            None.
        """
        idx = event.cursor_row
        if idx < len(self.clusters):
            self.selected_cluster = self.clusters[idx]
            log.info(
                "[tui] JoinApp cluster selected: %s (uid=%s)",
                self.selected_cluster.get("name", "?"),
                self.selected_cluster.get("uid", "?"),
            )
        self.exit()

    async def action_quit(self) -> None:
        """Handle the ``q`` keybinding: exit without selecting a cluster.

        Args:
            None.

        Returns:
            None.
        """
        log.info("[tui] JoinApp quit — no cluster selected")
        self.selected_cluster = None
        self.exit()


def format_elapsed(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string.

    Args:
        seconds: Elapsed seconds (can be fractional).

    Returns:
        String like ``"45s"``, ``"2m 30s"``, or ``"1h 15m"``.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class DashboardApp(App):
    COMMANDS = set()
    CSS = """
    Screen { background: $surface; layout: vertical; }
    #header { padding: 1 2 0 2; }
    #nodes { height: auto; max-height: 50%; margin: 0 1; }
    #stats { color: $text-muted; padding: 0 2; }
    #logs { min-height: 3; height: 1fr; margin: 0 1; }
    #footer { dock: bottom; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [Binding("q", "quit", "Quit"), Binding("l", "toggle_logs", "Logs")]

    def __init__(
        self,
        get_state: Callable,
        cluster_name: str,
        uid: str,
        my_rank: Optional[int] = None,
        log_capture: Optional["LogCapture"] = None,
        done_event: Optional["threading.Event"] = None,
        error_event: Optional["threading.Event"] = None,
    ) -> None:
        """Initialise the DashboardApp with cluster state and optional log capture.

        Args:
            get_state:   Zero-arg callable (or callable returning callable) that
                         returns the current cluster state dict.
            cluster_name: Human-readable cluster name shown in the header.
            uid:         Unique cluster identifier shown in the header.
            my_rank:     This node's rank, or ``None`` for the coordinator view.
            log_capture: Optional :class:`LogCapture` ring buffer; if provided,
                         log lines are displayed in the TUI log panel.
            done_event:  Optional event set when training completes successfully.
            error_event: Optional event set when training exits with an error.

        Returns:
            None.
        """
        super().__init__()
        self.get_state = get_state
        self.cluster_name = cluster_name
        self.uid = uid
        self.my_rank = my_rank
        self.log_capture = log_capture
        self.done_event = done_event
        self.error_event = error_event
        self.training_done = False
        self.logs_visible = True
        self.start_time = time.monotonic()

    def compose(self) -> ComposeResult:
        """Build the DashboardApp widget tree.

        Args:
            None.

        Returns:
            :class:`~textual.app.ComposeResult` yielding the header, node table,
            stats bar, log panel, and footer widgets.
        """
        role = f"rank {self.my_rank}" if self.my_rank is not None else "coordinator"
        yield Static(
            f"[b]grove[/]  {self.cluster_name}  [dim cyan]{self.uid}[/]  {role}",
            id="header",
        )
        table = DataTable(id="nodes")
        table.cursor_type = "none"
        table.add_columns(
            "Rank", "Host", "Status", "Step", "Loss", "Grad", "tok/s", "Sync", "Net Mb"
        )
        yield table
        yield Static("", id="stats")
        log = RichLog(id="logs", wrap=True, markup=False)
        log.show_vertical_scrollbar = False
        yield log
        yield Static("  [dim]q quit   l toggle logs[/]", id="footer")

    def on_mount(self) -> None:
        """Set up a 1-second refresh interval and trigger an initial render on mount.

        Args:
            None.

        Returns:
            None.
        """
        self.set_interval(1.0, self.refresh_dash)
        self.refresh_dash()

    def refresh_dash(self) -> None:
        """Refresh all three dashboard panels: header, node table, and log pane.

        Args:
            None.

        Returns:
            None.
        """
        self.refresh_header()
        self.refresh_table()
        self.refresh_logs()

    def refresh_header(self) -> None:
        """Update the header widget with current elapsed time and training status.

        Args:
            None.

        Returns:
            None.
        """
        elapsed = format_elapsed(time.monotonic() - self.start_time)
        role = f"rank {self.my_rank}" if self.my_rank is not None else "coordinator"
        if self.training_done:
            if self.error_event and self.error_event.is_set():
                self.query_one("#header", Static).update(
                    f"[bold red]exited with error[/]  {self.cluster_name}  [dim]{elapsed}[/]  press q to exit"
                )
            else:
                self.query_one("#header", Static).update(
                    f"[bold green]training complete[/]  {self.cluster_name}  [dim]{elapsed}[/]  press q to exit"
                )
        else:
            self.query_one("#header", Static).update(
                f"[b]grove[/]  {self.cluster_name}  [dim cyan]{self.uid}[/]  {role}  [dim]{elapsed}[/]"
            )
        if not self.training_done and self.done_event and self.done_event.is_set():
            self.training_done = True

    def refresh_table(self) -> None:
        """Repopulate the node table from the latest cluster state snapshot.

        Args:
            None.

        Returns:
            None.
        """
        state: Any = self.get_state()
        if not state:
            return
        if callable(state):
            state = state()

        table = self.query_one("#nodes", DataTable)
        table.clear()

        live_ranks = state.get("live_ranks", [])
        dead_ranks = state.get("dead_ranks", [])
        all_ranks = sorted(set(live_ranks) | set(dead_ranks))
        steps = state.get("steps", {})
        losses = state.get("loss", {})
        syncs = state.get("sync_ms", {})
        grad_norms = state.get("grad_norm", {})
        tok_per_secs = state.get("tok_per_sec", {})
        tx_mbps_map = state.get("tx_mbps", {})
        rx_mbps_map = state.get("rx_mbps", {})
        hostnames = state.get("hostnames", {})
        statuses = state.get("status", {})
        epoch = state.get("epoch", 0)

        def _get(d, rank):
            """Look up *rank* in *d* trying str, original, and int forms.

            Args:
                d:    Dict keyed by rank in any of str/int/original forms.
                rank: Rank value to look up.

            Returns:
                The matching value, or ``None`` if not found under any key form.
            """
            return d.get(str(rank), d.get(rank, d.get(int(rank), None)))

        for rank in all_ranks:
            hostname = _get(hostnames, rank) or f"node-{rank}"
            is_me = self.my_rank is not None and int(rank) == int(self.my_rank)
            is_dead = int(rank) in [int(d) for d in dead_ranks]

            if is_dead:
                step_val = _get(steps, rank) or "—"
                table.add_row(
                    Text(str(rank), style="dim"),
                    Text(str(hostname), style="dim"),
                    Text("dead", style="red"),
                    Text(str(step_val), style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                )
            else:
                step_val = _get(steps, rank) or 0
                loss_val = _get(losses, rank)
                sync_val = _get(syncs, rank)
                gn_val = _get(grad_norms, rank)
                tps_val = _get(tok_per_secs, rank)
                tx_val = _get(tx_mbps_map, rank)
                rx_val = _get(rx_mbps_map, rank)
                loss_str = f"{float(loss_val):.4f}" if loss_val else "—"
                sync_str = f"{float(sync_val):.0f}ms" if sync_val else "—"
                gn_str = f"{float(gn_val):.2f}" if gn_val else "—"
                tps_str = f"{float(tps_val):.0f}" if tps_val else "—"
                if tx_val and rx_val:
                    net_str = f"{float(tx_val):.0f}↑/{float(rx_val):.0f}↓"
                elif tx_val or rx_val:
                    net_str = f"{float(tx_val or rx_val):.0f}"
                else:
                    net_str = "—"
                style = "bold cyan" if is_me else ""
                marker = " ◀" if is_me else ""
                phase = _get(statuses, rank) or ""
                if phase and phase != "training":
                    status_text = Text(phase, style="yellow")
                else:
                    status_text = Text("ok", style="green")

                table.add_row(
                    Text(str(rank), style=style),
                    Text(str(hostname) + marker, style=style),
                    status_text,
                    Text(str(step_val) if step_val else "—", style=style),
                    Text(loss_str, style=style),
                    Text(gn_str, style=style),
                    Text(tps_str, style=style),
                    Text(sync_str, style=style),
                    Text(net_str, style=style),
                )

        n_live = len(live_ranks)
        n_dead = len(dead_ranks)

        parts = [f"  {n_live} node{'s' if n_live != 1 else ''}"]
        if n_dead:
            parts.append(f"  {n_dead} dead")
        if len(live_ranks) > 1:
            mesh = " ─ ".join(str(r) for r in live_ranks)
            parts.append(f"  {mesh}  (all-to-all)")
        if epoch:
            parts.append(f"  epoch {epoch}")

        self.query_one("#stats", Static).update(Text("".join(parts), style="dim"))

    def refresh_logs(self) -> None:
        """Drain new log lines from the capture buffer and append them to the log panel.

        Args:
            None.

        Returns:
            None.
        """
        if self.log_capture is None or not self.logs_visible:
            return
        log_widget = self.query_one("#logs", RichLog)
        for line in self.log_capture.drain_new():
            log_widget.write(Text.from_ansi(line))

    def action_toggle_logs(self) -> None:
        """Toggle the log panel visibility in response to the ``l`` keybinding.

        Args:
            None.

        Returns:
            None.
        """
        self.logs_visible = not self.logs_visible
        log_widget = self.query_one("#logs", RichLog)
        log_widget.display = self.logs_visible
        state = "on" if self.logs_visible else "off"
        self.query_one("#footer", Static).update(
            f"  [dim]q quit   l toggle logs ({state})[/]"
        )

    async def action_quit(self) -> None:
        """Handle the ``q`` keybinding: exit the dashboard.

        Args:
            None.

        Returns:
            None.
        """
        self.exit()


class WorkerPickerApp(App):
    """Select discovered smoltorrent workers to add to the cluster.

    Space toggles a row, a selects/deselects all, Enter confirms, q aborts.
    """

    COMMANDS = set()
    CSS = """
    Screen { background: $surface; }
    #title { padding: 1 2; }
    #hint { color: $text-muted; padding: 0 2 1 2; }
    DataTable { height: 1fr; }
    DataTable > .datatable--cursor { background: $accent; color: $text; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=False),
        Binding("space", "toggle_select", "Toggle", show=False),
        Binding("a", "select_all", "All", show=False),
        Binding("enter", "confirm", "Confirm", show=False, priority=True),
    ]

    def __init__(self, discovered: list[dict]) -> None:
        """Initialise the WorkerPickerApp with a list of discovered worker nodes.

        Args:
            discovered: List of worker dicts (rank, hostname, ip, port) from mDNS
                        or AirDrop discovery.

        Returns:
            None.
        """
        super().__init__()
        self.smolt_nodes = discovered
        self.selected: set[int] = set()
        self.chosen: list[dict] = []

    def compose(self) -> ComposeResult:
        """Build the WorkerPickerApp widget tree.

        Args:
            None.

        Returns:
            :class:`~textual.app.ComposeResult` yielding the title, worker table,
            and hint bar widgets.
        """
        yield Static("[b]smoltorrent[/]  select nodes to add to cluster", id="title")
        table = DataTable(id="workers")
        table.cursor_type = "row"
        table.add_columns("", "Rank", "Hostname", "IP", "Port")
        yield table
        yield Static(
            "  [dim]↑↓ move   space select   a all/none   enter confirm   q quit[/]",
            id="hint",
        )

    def on_mount(self) -> None:
        """Populate the worker table and focus it on mount.

        Args:
            None.

        Returns:
            None.
        """
        self.refresh_worker_table()
        self.query_one("#workers", DataTable).focus()

    def refresh_worker_table(self) -> None:
        """Repopulate the worker table, preserving the current cursor position.

        Args:
            None.

        Returns:
            None.
        """
        table = self.query_one("#workers", DataTable)
        cursor = table.cursor_row
        table.clear()
        for i, w in enumerate(self.smolt_nodes):
            checked = i in self.selected
            mark = Text("✓", style="green bold") if checked else Text(" ")
            style = "bold" if checked else ""
            table.add_row(
                mark,
                Text(str(w["rank"]), style=style),
                Text(w["hostname"], style=style),
                Text(w["ip"], style="cyan" if checked else "dim"),
                Text(str(w["port"]), style=style),
            )
        if cursor is not None and self.smolt_nodes:
            table.move_cursor(row=min(cursor, len(self.smolt_nodes) - 1))

    def action_toggle_select(self) -> None:
        """Toggle the selection state of the row under the cursor.

        Args:
            None.

        Returns:
            None.
        """
        idx = self.query_one("#workers", DataTable).cursor_row
        if idx in self.selected:
            self.selected.discard(idx)
        else:
            self.selected.add(idx)
        self.refresh_worker_table()

    def action_select_all(self) -> None:
        """Toggle between selecting all workers and clearing the selection.

        Args:
            None.

        Returns:
            None.
        """
        if len(self.selected) == len(self.smolt_nodes):
            self.selected.clear()
        else:
            self.selected = set(range(len(self.smolt_nodes)))
        self.refresh_worker_table()

    def action_confirm(self) -> None:
        """Store the selected workers in ``chosen`` and exit the app.

        Args:
            None.

        Returns:
            None.
        """
        self.chosen = [self.smolt_nodes[i] for i in sorted(self.selected)]
        self.exit()

    async def action_quit(self) -> None:
        """Handle the ``q`` keybinding: exit without confirming any selection.

        Args:
            None.

        Returns:
            None.
        """
        self.chosen = []
        self.exit()
