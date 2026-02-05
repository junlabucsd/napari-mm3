from dataclasses import dataclass
from pathlib import Path

import numpy as np
from magicgui.widgets import FileEdit, LineEdit, PushButton, SpinBox
from napari import Viewer

from ._deriving_widgets import (
    MM3Container2,
    SegmentationMode,
    load_specs,
    load_tiff,
)
from .utils import (
    TIFF_FILE_FORMAT_PEAK,
    Cell,
    Cells,
    read_cells_from_json,
    write_cells_to_json,
    write_cells_to_matlab,
)

TRANSLUCENT_BLUE = np.array([0.0, 0.0, 1.0, 1.0])
VERY_TRANSLUCENT_BLUE = np.array([0.0, 0.0, 1.0, 0.3])
TRANSLUCENT_GREEN = np.array([0.0, 1.0, 0.0, 0.3])
TRANSLUCENT_RED = np.array([1.0, 0.0, 0.0, 1.0])
VERY_TRANSLUCENT_RED = np.array([1.0, 0.5, 0.0, 0.3])
TRANSPARENT = np.array([0, 0, 0, 0])


def calc_cell_list_times(cells: Cells, cell_list):
    times = set()
    for cell in cell_list:
        add_times = set(cells[cell].times.copy())
        add_times = set(map(lambda t: t, add_times))
        times = times.union(add_times)
    out = list(times)
    out.sort()
    return out


def get_cell_lineage(cell_id: str, cells: Cells, cell_gens):
    """Given a cell id, a 'cells' dictionary, and the lineage length, returns:
    * the lineage with cell_gens cell_ids representing where the cell came from.
    * None: if such a lineage goes too far back.
    """
    cell_lineage = [cell_id]
    for _ in range(cell_gens - 1):
        last_cell_id = cell_lineage[-1]
        last_cell_info = cells[last_cell_id]
        parent_id = last_cell_info.parent
        if parent_id is None:
            return cell_lineage
        cell_lineage.append(parent_id)
    return cell_lineage


def cell_iter(complete_cell_ids: list, all_cells: Cells, generations, min_gens):
    """
    Iterates over every complete cell that satisfies certain criteria, and yields
    info relevant to visualization.
    It must have a lineage
    """
    for cell_id in complete_cell_ids:
        lineage = get_cell_lineage(cell_id, all_cells, cell_gens=generations)
        if len(set(lineage)) < min_gens:
            continue
        cur_time_range = calc_cell_list_times(all_cells, lineage)
        yield cell_id, list(lineage)  # , min(cur_time_range), max(cur_time_range)


def gen_cell_list(complete_cell_ids: list, all_cells: Cells, generations, min_gens):
    """
    Creates a list of every cell that satisfies certain criteria.
    """
    cell_ids = []
    for cell_id in complete_cell_ids:
        lineage = get_cell_lineage(cell_id, all_cells, cell_gens=generations)
        if len(set(lineage)) < min_gens:
            continue
        cell_ids.append(cell_id)
    return cell_ids


@dataclass
class InPaths:
    experiment_name: str = ""
    all_cells_path: Path = Path("./analysis/cell_data/all_cells.json")
    complete_cells_path: Path = Path("./analysis/cell_data/complete_cells.json")
    replication_cells_path: Path = Path(
        "./analysis/cell_data/replication_cells.json"
    )  # Technically this is an output path!
    segmented_path: Path = Path("./analysis/segmented")
    analysis_folder: Path = Path("./analysis")
    segmentation_mode: SegmentationMode = SegmentationMode.OTSU


class FociPicking(MM3Container2):
    """
    Note to reader:
    This is just *barely* illegible. If you find yourself here, please do a bit of cleanup work!

    Ok, goal here is to make this load only AFTER a cells file is specified.
    This implies two states:
    1. Before specified
    2. After specified.
    """

    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.viewer.grid.enabled = False
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.color = "white"

        self.in_paths = InPaths()
        self.add_in_folders()

        self.num_generations = 2
        self.min_generations = self.num_generations
        self.seg_visible = True

        self.loaded_preview = False
        self.load_preview_widget = PushButton(label="load interactive")
        self.append(self.load_preview_widget)
        self.load_preview_widget.changed.connect(self.load_preview)

    def make_widgets(self):
        self.viewer.bind_key("a", self.next_cell)
        self.viewer.bind_key("s", self.prev_cell)

        self.next_cell_widget = PushButton(
            label="next cell", tooltip="Can also get there by pressing 's'."
        )
        self.prev_cell_widget = PushButton(
            label="prev cell", tooltip="Can also get there by pressing 'a'."
        )
        self.crop_left_widget = SpinBox(
            label="left_crop", min=0, max=self.crop_right, value=self.crop_left
        )
        self.crop_right_widget = SpinBox(
            label="right_crop", min=0, max=self.crop_right, value=self.crop_right
        )
        self.cell_min_generations_widget = SpinBox(
            label="min_generations", min=1, max=5, value=self.num_generations
        )
        self.cell_generations_widget = SpinBox(
            label="generations", min=1, max=5, value=self.num_generations
        )
        self.jump_to_cell_id_widget = LineEdit(label="skip_to_cell_id")
        # cell label is the position of the cell within the trench (min == mother cell)
        self.cell_label_widget = SpinBox(label="cell_label", min=1, max=5, value=1)
        self.save_to_matlab_widget = PushButton(label="save_to_matlab")

        self.append(self.next_cell_widget)
        self.append(self.prev_cell_widget)
        self.append(self.crop_left_widget)
        self.append(self.crop_right_widget)
        self.append(self.cell_min_generations_widget)
        self.append(self.cell_generations_widget)
        self.append(self.cell_label_widget)
        self.append(self.jump_to_cell_id_widget)
        self.append(self.save_to_matlab_widget)

        self.next_cell_widget.clicked.connect(self.next_cell)
        self.prev_cell_widget.clicked.connect(self.prev_cell)
        self.crop_left_widget.changed.connect(self.set_crop_left)
        self.crop_left_widget.changed.connect(self.update_preview)
        self.crop_right_widget.changed.connect(self.set_crop_right)
        self.crop_right_widget.changed.connect(self.update_preview)
        self.cell_min_generations_widget.changed.connect(self.set_cell_generations)
        self.cell_generations_widget.changed.connect(self.set_cell_generations)
        self.cell_label_widget.changed.connect(self.cell_label_changed)
        self.save_to_matlab_widget.changed.connect(self.save_to_matlab)
        self.jump_to_cell_id_widget.changed.connect(self.jump_to_cell_id)

    def load_preview(self):
        self.loaded_preview = True
        self.all_cells = Cells(read_cells_from_json(self.in_paths.all_cells_path))

        # load cells with replication data, or make a new dict if no previous results exist.
        self.replication_cells = {}
        if self.in_paths.replication_cells_path.exists():
            self.replication_cells = Cells(
                read_cells_from_json(self.in_paths.replication_cells_path)
            )
            for cell_id, cell in self.replication_cells.items():
                self.all_cells[cell_id] = cell

        json_cells = Cells(read_cells_from_json(self.in_paths.complete_cells_path))
        specs = load_specs(self.in_paths.analysis_folder)
        self.mapping = self.all_cells.gen_label_to_cell_mapping(specs)

        self.mother_cells = {}
        for cell_id in json_cells:
            if json_cells[cell_id].birth_label == 1:
                self.mother_cells[cell_id] = json_cells[cell_id]
        self.cell_lineages = list(
            cell_iter(
                complete_cell_ids=self.mother_cells.keys(),
                all_cells=self.all_cells,
                generations=self.num_generations,
                min_gens=self.min_generations,
            )
        )
        cell_id, cur_lineage = self.cell_lineages[0]
        self.cur_lineage = cur_lineage
        print(cur_lineage)

        self.cell_idx = 0
        self.cell_label = 1
        self.update_cell_info()

        stack_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            self.fov_id,
            self.peak_id,
            "c1",
        )
        stack = load_tiff(self.in_paths.analysis_folder / "channels" / stack_filename)

        self.im_height = stack.shape[1]
        self.crop_left = 0
        self.crop_right = stack.shape[2] - 1

        self.make_widgets()
        self.update_preview()

    def update_cell_info(self):
        """Relies on: self.cell_lineages, self.cell_idx, self.all_cells"""
        self.cur_cell_id, self.cur_lineage = self.cell_lineages[self.cell_idx]
        self.cur_cell: Cell = self.all_cells[self.cur_cell_id]
        cur_time_range = calc_cell_list_times(self.all_cells, self.cur_lineage)
        self.start = min(cur_time_range) + 1
        self.stop = max(cur_time_range) + 1
        self.fov_id = self.cur_cell.fov
        self.peak_id = self.cur_cell.peak
        if not hasattr(self.cur_cell, "initiations"):
            self.cur_cell.initiations = []
            self.cur_cell.initiation_cells = []
        if not hasattr(self.cur_cell, "terminations"):
            self.cur_cell.terminations = []
            self.cur_cell.termination_cells = []

    def update_preview(self):
        self.viewer.text_overlay.text = (
            f"Cell idx: {self.cell_idx + 1} / {len(self.cell_lineages)}\n"
            f"Cell ID: {self.cur_cell_id}"
        )

        self.seg_visible = True
        if "segmentation" in self.viewer.layers:
            self.seg_visible = self.viewer.layers["segmentation"].visible
        self.viewer.layers.clear()
        print(f"showing cell {self.cur_cell_id}")

        # switch postfix to c1/c2/c3 auto??
        stack_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            self.fov_id,
            self.peak_id,
            "c1",
        )
        stack = load_tiff(self.in_paths.analysis_folder / "channels" / stack_filename)

        stack_fl_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            self.fov_id,
            self.peak_id,
            "c2",
        )
        stack_fl = load_tiff(
            self.in_paths.analysis_folder / "channels" / stack_fl_filename
        )

        stack_filtered = stack[
            self.start - 1 : self.stop, :, self.crop_left : self.crop_right + 1
        ]
        stack_fl_filtered = stack_fl[
            self.start - 1 : self.stop, :, self.crop_left : self.crop_right + 1
        ]
        stack_filtered = np.concatenate(stack_filtered, axis=1)
        stack_fl_filtered = np.concatenate(stack_fl_filtered, axis=1)

        final_image = np.stack((stack_filtered, stack_fl_filtered))
        images = self.viewer.add_image(final_image, name="image")
        self.vis_seg_stack()
        images.reset_contrast_limits()

        if "time_labels" in self.viewer.layers:
            self.viewer.layers.remove("time_labels")
        visible_times = np.arange(self.start, self.stop + 1)
        features = {"time_label": visible_times}
        pt_xs = (np.arange(len(visible_times)) + 0.5) * (
            self.crop_right - self.crop_left + 1
        )
        pt_ys = np.zeros(visible_times.shape)
        pts = np.stack((pt_ys, pt_xs), axis=1)
        text = {"string": "{time_label}", "color": "white", "size": 8}
        self.viewer.add_points(
            pts,
            text=text,
            features=features,
            face_color=TRANSPARENT,
            edge_color=TRANSPARENT,
            name="time_labels",
        )

        if hasattr(self.cur_cell, "initiations"):
            self.vis_initiations()

        if hasattr(self.cur_cell, "terminations"):
            self.vis_terminations()

        cur_step = list(self.viewer.dims.current_step)
        cur_step[0] = 1
        self.viewer.dims.current_step = tuple(cur_step)
        self.viewer.layers.selection.clear()
        self.viewer.layers["image"].reset_contrast_limits()

        self.viewer.layers.selection.update({self.viewer.layers["image"]})
        self.viewer.layers["image"].mouse_drag_callbacks.append(self.click_callback)

    def vis_seg_stack(self):
        seg_str = (
            "seg_otsu"
            if self.in_paths.segmentation_mode == SegmentationMode.OTSU
            else "seg_unet"
        )
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            self.fov_id,
            self.peak_id,
            seg_str,
        )
        seg_stack = load_tiff(self.in_paths.segmented_path / img_filename)

        seg_stack = seg_stack[
            self.start - 1 : self.stop, :, self.crop_left : self.crop_right + 1
        ]
        new_seg_stack = np.full(seg_stack.shape, False)
        for cell_id in self.cur_lineage:
            cell = self.all_cells[cell_id]
            for t, label in zip(cell.times, cell.labels):
                t_ = t - self.start
                new_seg_stack[t_] = new_seg_stack[t_] | (seg_stack[t_] == label)

        init_shift_stack = np.zeros(seg_stack.shape, dtype=np.int64)
        if hasattr(self.cur_cell, "initiations"):
            for init_cell_id, init_time in zip(
                self.cur_cell.initiation_cells, self.cur_cell.initiations
            ):
                vis_time = round(init_time) - self.start
                init_cell = self.all_cells[init_cell_id]
                cell_time_idx = init_cell.times.index(init_time)
                cell_label = init_cell.labels[cell_time_idx]
                init_shift_stack[vis_time, :, :] += seg_stack[vis_time] == cell_label

        if hasattr(self.cur_cell, "terminations"):
            term_shift_stack = np.zeros(seg_stack.shape, dtype=np.int64)
            for term_cell_id, term_time in zip(
                self.cur_cell.termination_cells, self.cur_cell.terminations
            ):
                vis_time = round(term_time) - self.start
                term_cell = self.all_cells[term_cell_id]
                cell_time_idx = term_cell.times.index(term_time)
                cell_label = term_cell.labels[cell_time_idx]
                term_shift_stack[vis_time, :, :] += 9 * (
                    seg_stack[vis_time] == cell_label
                )

            new_seg_stack = (
                new_seg_stack.astype(np.int64) + init_shift_stack + term_shift_stack
            )

        new_seg_stack = np.concatenate(new_seg_stack, axis=1)
        if "segmentation" in self.viewer.layers:
            self.seg_visible = self.viewer.layers["segmentation"].visible
            self.viewer.layers.remove("segmentation")
        self.viewer.add_labels(new_seg_stack, name="segmentation")
        self.viewer.layers["segmentation"].visible = self.seg_visible

        self.viewer.layers.selection.update({self.viewer.layers[-1]})
        self.viewer.layers[-1].mouse_drag_callbacks.append(self.click_callback)
        self.viewer.layers[-1].mouse_double_click_callbacks.append(self.click_callback)

    def vis_terminations(self):
        if "terminations" in self.viewer.layers:
            self.viewer.layers.remove("terminations")
        shapes = self.viewer.add_shapes(name="terminations")
        termination_times = self.cur_cell.terminations
        for termination in set(termination_times):
            rel_init = termination - self.start
            left_bdry = rel_init * (self.crop_right + 1 - self.crop_left)
            right_bdry = (rel_init + 1) * (self.crop_right + 1 - self.crop_left)
            if termination_times.count(termination) == 2:
                shapes.add_rectangles(
                    [[0, left_bdry], [self.im_height, right_bdry]],
                    edge_color=TRANSLUCENT_RED,
                    face_color=VERY_TRANSLUCENT_RED,
                    edge_width=3,
                )
                continue
            if termination_times.count(termination) > 2:
                shapes.add_rectangles(
                    [[0, left_bdry], [self.im_height, right_bdry]],
                    edge_color=TRANSLUCENT_RED,
                    face_color=VERY_TRANSLUCENT_RED,
                    edge_width=3 * termination_times.count(termination),
                )
                continue
            shapes.add_rectangles(
                [[0, left_bdry], [self.im_height, right_bdry]],
                edge_color=TRANSLUCENT_RED,
                face_color=TRANSPARENT,
                edge_width=3,
            )

    def vis_initiations(self):
        if "initiations" in self.viewer.layers:
            self.viewer.layers.remove("initiations")
        shapes = self.viewer.add_shapes(name="initiations")
        initiation_times = self.cur_cell.initiations
        for initiation in set(initiation_times):
            rel_init = initiation - self.start
            left_bdry = rel_init * (self.crop_right + 1 - self.crop_left)
            right_bdry = (rel_init + 1) * (self.crop_right + 1 - self.crop_left)
            if initiation_times.count(initiation) == 2:
                shapes.add_rectangles(
                    [[0, left_bdry], [self.im_height, right_bdry]],
                    edge_color=TRANSLUCENT_BLUE,
                    face_color=TRANSLUCENT_GREEN,
                    edge_width=3,
                )
                continue
            if initiation_times.count(initiation) > 2:
                shapes.add_rectangles(
                    [[0, left_bdry], [self.im_height, right_bdry]],
                    edge_color=TRANSLUCENT_GREEN,
                    face_color=VERY_TRANSLUCENT_BLUE,
                    edge_width=3 * initiation_times.count(initiation),
                )
                continue
            shapes.add_rectangles(
                [[0, left_bdry], [self.im_height, right_bdry]],
                edge_color=TRANSLUCENT_BLUE,
                face_color=TRANSPARENT,
                edge_width=3,
            )

    def mark_initiation(self, viewer: Viewer):
        t, x, y = self.cursor_coords()
        init_cell = self.first_lineage_cell(t)
        if init_cell:
            self.cur_cell.initiations.append(t)
            self.cur_cell.initiation_cells.append(init_cell)
            self.replication_cells[self.cur_cell_id] = self.cur_cell
            self.vis_initiations()
            self.vis_seg_stack()

    def remove_termination(self, viewer: Viewer):
        t, x, y = self.cursor_coords()
        try:
            idx = self.cur_cell.terminations.index(t)
            self.cur_cell.terminations.remove(t)
            self.cur_cell.termination_cells.pop(idx)
        except ValueError:
            print("WARNING: tried to remove an initiation that does not exist.")
        self.replication_cells[self.cur_cell_id] = self.cur_cell
        self.vis_terminations()
        self.vis_seg_stack()

    def remove_initiation(self, viewer: Viewer):
        t, x, y = self.cursor_coords()
        try:
            idx = self.cur_cell.initiations.index(t)
            self.cur_cell.initiations.remove(t)
            self.cur_cell.initiation_cells.pop(idx)
        except ValueError:
            print("WARNING: tried to remove an initiation that does not exist.")
        self.replication_cells[self.cur_cell_id] = self.cur_cell
        self.vis_initiations()
        self.vis_seg_stack()

    def mark_termination(self, viewer: Viewer):
        t, x, y = self.cursor_coords()
        term_cell = self.first_lineage_cell(t)
        if term_cell:
            self.cur_cell.terminations.append(t)
            self.cur_cell.termination_cells.append(term_cell)
            self.replication_cells[self.cur_cell_id] = self.cur_cell
            self.vis_terminations()
            self.vis_seg_stack()

    def locate_cell_at_cursor(self):
        t, x, y = self.cursor_coords()
        seg_str = (
            "seg_otsu"
            if self.in_paths.segmentation_mode == SegmentationMode.OTSU
            else "seg_unet"
        )
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            self.fov_id,
            self.peak_id,
            seg_str,
        )

        seg_stack = load_tiff(self.analysis_folder / "segmented" / img_filename)
        cur_seg_stack = seg_stack[t, :, self.crop_left : self.crop_right + 1]
        # TODO: Proper rounding.
        cur_label = cur_seg_stack[y, x]
        # need to cache this later!
        if cur_label == 0:
            return None
        try:
            return self.mapping[self.fov_id][self.peak_id][t][cur_label]
        except KeyError:
            return None

    def set_cell_generations(self):
        self.num_generations = self.cell_generations_widget.value
        self.min_generations = self.cell_min_generations_widget.value
        cell_lineage_iter = cell_iter(
            complete_cell_ids=self.mother_cells.keys(),
            all_cells=self.all_cells,
            generations=self.num_generations,
            min_gens=self.min_generations,
        )
        self.cell_lineages = list(cell_lineage_iter)
        self.cell_idx = 0
        self.cur_cell_id, self.cur_lineage = self.cell_lineages[self.cell_idx]
        cur_time_range = calc_cell_list_times(self.all_cells, self.cur_lineage)
        self.start = min(cur_time_range) + 1
        self.stop = max(cur_time_range) + 1
        self.fov_id = self.all_cells[self.cur_cell_id].fov
        self.peak_id = self.all_cells[self.cur_cell_id].peak
        self.update_preview()

    def cursor_coords(self):
        coords = self.viewer.cursor.position
        timestamp = (
            round(coords[2] // (self.crop_right + 1 - self.crop_left)) + self.start
        )
        timestamp_clamped = min(max(self.start, timestamp), self.stop)
        x_coord = round(coords[2]) % (self.crop_right + 1 - self.crop_left)
        y_coord = round(coords[1])
        return timestamp_clamped, x_coord, y_coord

    def save_to_matlab(self):
        # This prevents fun side effects with editing the various cell dictionaries.
        old_cells = read_cells_from_json(self.replication_cell_loc)
        write_cells_to_matlab(
            old_cells, self.analysis_folder / "cell_data" / "replication_cells.mat"
        )
        print("save to matlab")

    def skip(self, viewer: Viewer):
        if hasattr(self.cur_cell, "initiations"):
            delattr(self.cur_cell, "initiations")
        if hasattr(self.cur_cell, "initiation_cells"):
            delattr(self.cur_cell, "initiation_cells")
        if hasattr(self.cur_cell, "terminations"):
            delattr(self.cur_cell, "terminations")
        if hasattr(self.cur_cell, "termination_cells"):
            delattr(self.cur_cell, "termination_cells")
        self.cell_idx = min(self.cell_idx + 1, len(self.cell_lineages) - 1)
        self.update_cell_info()
        self.update_preview()

    def first_lineage_cell(self, t):
        """Given a time, gets the first cell from the lineage that is visible at
        that time"""
        available_labels = self.mapping[self.fov_id][self.peak_id][t]
        for label, labelled_cell_id in available_labels.items():
            if labelled_cell_id in self.cur_lineage:
                return labelled_cell_id
        return None

    def update_lineages(self, cells):
        self.mother_cells = {}
        for cell_id in cells:
            if cells[cell_id].birth_label == self.cell_label:
                self.mother_cells[cell_id] = cells[cell_id]

        cell_lineage_iter = cell_iter(
            complete_cell_ids=self.mother_cells.keys(),
            all_cells=self.all_cells,
            generations=self.num_generations,
            min_gens=self.min_generations,
        )
        self.cell_lineages = list(cell_lineage_iter)
        self.cell_idx = 0
        self.update_cell_info()
        self.update_preview()

    def cell_label_changed(self):
        self.cell_label = self.cell_label_widget.value
        self.update_lineages(Cells(read_cells_from_json(self.cell_json_loc)))

    def click_callback(self, layer, event):
        coords = self.viewer.cursor.position
        y_coord = round(coords[1])
        timestamp = (
            round(coords[2] // (self.crop_right + 1 - self.crop_left)) + self.start
        )
        if (timestamp < 0) or (y_coord < 0):
            return
        if (timestamp > self.stop) or (y_coord > self.im_height):
            return

        # left click => initiation
        if event.button == 1:
            self.mark_initiation(self.viewer)
        # middle click => delete current
        if event.button == 3:
            self.remove_initiation(self.viewer)
            self.remove_termination(self.viewer)
        # right click => termination
        if event.button == 2:
            self.mark_termination(self.viewer)

    def set_crop_left(self):
        self.crop_left = self.crop_left_widget.value

    def set_crop_right(self):
        self.crop_right = self.crop_right_widget.value

    def next_cell(self, viewer: Viewer):
        # TODO: not sure if we need the viewer parameter.
        write_cells_to_json(
            self.replication_cells, self.in_paths.replication_cells_path
        )
        self.cell_idx = min(self.cell_idx + 1, len(self.cell_lineages) - 1)
        self.update_cell_info()
        self.update_preview()

    def prev_cell(self, viewer: Viewer):
        write_cells_to_json(
            self.replication_cells, self.in_paths.replication_cells_path
        )
        self.cell_idx = max(0, self.cell_idx - 1)
        self.update_cell_info()
        self.update_preview()

    def jump_to_cell_id(self):
        cell_id = self.jump_to_cell_id_widget.value
        write_cells_to_json(self.replication_cells, self.replication_cell_loc)
        cell_ids = [cell_id for cell_id, _ in self.cell_lineages]
        cell_idx = cell_ids.index(cell_id)

        self.cell_idx = cell_idx
        self.update_cell_info()
        self.update_preview()

    def toggle_seg_visibility(self, viewer):
        if "segmentation" in self.viewer.layers:
            self.viewer.layers["segmentation"].visible = not self.viewer.layers[
                "segmentation"
            ].visible
