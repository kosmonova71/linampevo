"""LinAmp application package."""

from __future__ import annotations


def _create_labeled_icon_button(text: str, icon_name: str | None = None):
    """Create a Gtk.Button with an optional leading icon (lazy GTK import)."""
    from .ui.widgets import create_icon_button

    return create_icon_button(text, icon_name)


def _create_modal_form(title, parent, field_labels, ok_label):
    """Build a small modal dialog with one Gtk.Entry per label.

    Returns ``(dialog, entries, cancel_button, ok_button)``.
    """
    from .deps import Gtk

    dialog = Gtk.Dialog(title=title, transient_for=parent, modal=True)
    dialog.set_default_size(420, -1)

    content = dialog.get_content_area()
    content.set_spacing(8)
    content.set_margin_top(12)
    content.set_margin_bottom(12)
    content.set_margin_start(12)
    content.set_margin_end(12)

    grid = Gtk.Grid()
    grid.set_row_spacing(8)
    grid.set_column_spacing(8)
    content.append(grid)

    entries = []
    for row, label_text in enumerate(field_labels):
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        grid.attach(label, 0, row, 1, 1)
        entry = Gtk.Entry()
        entry.set_hexpand(True)
        grid.attach(entry, 1, row, 1, 1)
        entries.append(entry)

    cancel_button = dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
    ok_button = dialog.add_button(ok_label, Gtk.ResponseType.OK)
    dialog.set_default_response(Gtk.ResponseType.OK)
    return dialog, entries, cancel_button, ok_button
