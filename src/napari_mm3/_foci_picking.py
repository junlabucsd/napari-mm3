import numpy as np

from napari import Viewer
from .utils import Cells, read_cells_from_json, write_cells_to_json, write_cells_to_matlab
from magicgui.widgets import SpinBox, PushButton
from ._deriving_widgets import (
    MM3Container,
    load_subtracted_stack,
    load_seg_stack,
    load_specs,
    SegmentationMode,
)

TRANSLUCENT_BLUE = np.array([0.0, 0.0, 1.0, 1.0])
TRANSLUCENT_RED = np.array([1.0, 0.0, 0.0, 1.0])
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
    """Given a cell id, a 'cells' dictionary, and the lineage length,
    returns:
        * the lineage with cell_gens cell_ids representing where the cell came from.
        * None: if such a lineage goes too far back.
    """
    cell_lineage = [cell_id]
    for i in range(cell_gens - 1):
        last_cell_id = cell_lineage[-1]
        last_cell_info = cells[last_cell_id]
        parent_id = last_cell_info.parent
        if parent_id is None:
            return None
        cell_lineage.append(parent_id)
    return cell_lineage


def cell_lineage_filter(complete_cell_ids: list, all_cells: Cells, generations):
    for cell_id in complete_cell_ids:
        lineage = get_cell_lineage(cell_id, all_cells, cell_gens=generations)
        if lineage is None:
            continue
        yield cell_id, lineage


class FociPicking(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""

        # tweakables:
        # x-crop-left (1)
        # x-crop-right (1)
        # # of gens visible
        # which cell u see trace for.
        # need to think about how to best encode the STATE of this widget -- this might generalize good.
        # maybe use magic widgets better??
        # TODO: Cleanup pass... maybe.

        self.experiment_name_widget.hide()
        self.experiment_name = "20220331_ALO7931_ALO7918_ABT"
        self.load_recent_widget.hide()
        self.run_widget.hide()

        self.viewer.grid.enabled = False

        # load a list of all complete cells.
        self.cell_file_loc = self.analysis_folder / "cell_data" / "all_cells.json"

        self.all_cells = Cells(read_cells_from_json(self.cell_file_loc))
        complete_cells = self.all_cells.find_complete_cells()
        minimal_timestamp = 10
        self.mother_cells = {}
        for cell_id in complete_cells:
            if complete_cells[cell_id].birth_label == 1:
                self.mother_cells[cell_id] = complete_cells[cell_id]
            minimal_timestamp = min(
                minimal_timestamp, min(complete_cells[cell_id].times)
            )
        print("min timestamp " + str(minimal_timestamp))

        self.num_generations = 2
        cell_lineage_iter = cell_lineage_filter(
            complete_cell_ids=self.mother_cells.keys(),
            all_cells=self.all_cells,
            generations=self.num_generations,
        )
        self.cell_lineages = list(cell_lineage_iter)
        self.cell_idx = 0
        self.update_cell_info()

        stack = load_subtracted_stack(
            self.analysis_folder,
            self.experiment_name,
            self.fov_id,
            self.peak_id,
            "sub_c1",
        )
        self.im_height = stack.shape[1]
        self.crop_left = 0
        self.crop_right = stack.shape[2] - 1

        self.crop_left_widget = SpinBox(
            label="left_crop", min=0, max=self.crop_right, value=self.crop_left
        )
        self.crop_right_widget = SpinBox(
            label="right_crop", min=0, max=self.crop_right, value=self.crop_right
        )
        self.cell_generations_widget = SpinBox(
            label="generations", min=1, max=5, value=self.num_generations
        )
        self.save_to_matlab_widget = PushButton(label="save_to_matlab")

        self.append(self.crop_left_widget)
        self.append(self.crop_right_widget)
        self.append(self.cell_generations_widget)
        self.append(self.save_to_matlab_widget)

        self.crop_left_widget.changed.connect(self.set_crop_left)
        self.crop_right_widget.changed.connect(self.set_crop_right)
        self.cell_generations_widget.changed.connect(self.set_cell_generations)
        self.save_to_matlab_widget.changed.connect(self.save_to_matlab)

        self.viewer.text_overlay.text = f"Cell idx: {self.cell_idx} / {len(self.cell_lineages)}\n"\
            f"Cell ID: {self.cur_cell_id}"
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.color = "white"

        self.viewer.bind_key("z", self.mark_initiation)
        self.viewer.bind_key("x", self.mark_termination)
        self.viewer.bind_key("s", self.remove_initiation)
        self.viewer.bind_key("c", self.prev_cell)
        self.viewer.bind_key("v", self.next_cell)
        self.update_preview()

    def update_cell_info(self):
        """Relies on: self.cell_lineages, self.cell_idx, self.all_cells"""
        self.cur_cell_id, self.cur_lineage = self.cell_lineages[self.cell_idx]
        self.cur_cell = self.all_cells[self.cur_cell_id]
        cur_time_range = calc_cell_list_times(self.all_cells, self.cur_lineage)
        self.start = min(cur_time_range)
        self.stop = max(cur_time_range)
        self.fov_id = self.cur_cell.fov
        self.peak_id = self.cur_cell.peak
        if not hasattr(self.cur_cell, "initiation"):
            self.cur_cell.initiation = []
            self.cur_cell.initiation_cells = []
        if not hasattr(self.cur_cell, "termination"):
            self.cur_cell.termination = None
            self.cur_cell.termination_cell = None

    def update_preview(self):
        self.viewer.layers.clear()
        print("showing cell" + str(self.cur_cell_id))

        # To do this properly, load all possible stacks.
        stack = load_subtracted_stack(
            self.analysis_folder,
            self.experiment_name,
            self.fov_id,
            self.peak_id,
            "sub_c1",
        )
        stack_fl = load_subtracted_stack(
            self.analysis_folder,
            self.experiment_name,
            self.fov_id,
            self.peak_id,
            "sub_c2",
        )
        stack_filtered = stack[
            self.start - 1:self.stop, :, self.crop_left:self.crop_right + 1
        ]
        stack_fl_filtered = stack_fl[
            self.start - 1:self.stop, :, self.crop_left:self.crop_right + 1
        ]
        stack_filtered = np.concatenate(stack_filtered, axis=1)
        stack_fl_filtered = np.concatenate(stack_fl_filtered, axis=1)

        final_image = np.stack((stack_filtered, stack_fl_filtered))
        images = self.viewer.add_image(final_image)
        self.vis_seg_stack()
        images.reset_contrast_limits()

        if self.cur_cell.initiation != []:
            self.vis_initiations()

        if self.cur_cell.termination is not None:
            self.vis_terminal()

    def vis_seg_stack(self):
        seg_stack = load_seg_stack(
            ana_dir=self.analysis_folder,
            experiment_name=self.experiment_name,
            fov_id=self.fov_id,
            peak_id=self.peak_id,
            seg_mode=SegmentationMode.UNET,
        )
        seg_stack = seg_stack[
            self.start - 1:self.stop, :, self.crop_left:self.crop_right + 1
        ]
        new_seg_stack = np.full(seg_stack.shape, False)
        for cell_id in self.cur_lineage:
            cell = self.all_cells[cell_id]
            for t, label in zip(cell.times, cell.labels):
                t_ = t - self.start
                new_seg_stack[t_] = new_seg_stack[t_] | (seg_stack[t_] == label)

        init_shift_stack = np.zeros(seg_stack.shape, dtype=np.int64)
        for init_cell_id, init_time in zip(
            self.cur_cell.initiation_cells, self.cur_cell.initiation
        ):
            vis_time = round(init_time) - self.start
            init_cell = self.all_cells[init_cell_id]
            cell_time_idx = init_cell.times.index(init_time)
            cell_label = init_cell.labels[cell_time_idx]
            init_shift_stack[vis_time, :, :] += seg_stack[vis_time] == cell_label

        term_shift_stack = np.zeros(seg_stack.shape, dtype=np.int64)
        if self.cur_cell.termination is not None:
            vis_term_time = round(self.cur_cell.termination) - self.start
            term_cell = self.all_cells[self.cur_cell.termination_cell]
            actual_term_time_idx = term_cell.times.index(self.cur_cell.termination)
            cell_region = term_cell.labels[actual_term_time_idx]
            term_shift_stack[vis_term_time, :, :] += 9 * (
                seg_stack[vis_term_time] == cell_region
            )

        new_seg_stack = (
            new_seg_stack.astype(np.int64) + init_shift_stack + term_shift_stack
        )
        new_seg_stack = np.concatenate(new_seg_stack, axis=1)
        if "segmentation" in self.viewer.layers:
            self.viewer.layers.remove("segmentation")
        self.viewer.add_labels(new_seg_stack, name="segmentation")

    def next_cell(self, viewer: Viewer):
        write_cells_to_json(self.all_cells, self.cell_file_loc)
        # write_cells_to_matlab(self.all_cells, self.analysis_folder / "cell_data" / "cell_data_foci.mat")
        self.cell_idx = min(self.cell_idx + 1, len(self.cell_lineages) - 1)
        self.update_cell_info()
        self.update_preview()

    def prev_cell(self, viewer: Viewer):
        write_cells_to_json(self.all_cells, self.cell_file_loc)
        # write_cells_to_matlab(self.all_cells, self.analysis_folder / "cell_data" / "cell_data_foci.mat")
        self.cell_idx = max(0, self.cell_idx - 1)
        self.update_cell_info()
        self.update_preview()

    def vis_terminal(self):
        if "termination" in self.viewer.layers:
            self.viewer.layers.remove("termination")
        shapes = self.viewer.add_shapes(name="termination")
        terminal_idx = self.cur_cell.termination - self.start
        left_bdry = terminal_idx * (self.crop_right + 1 - self.crop_left)
        right_bdry = (terminal_idx + 1) * (self.crop_right + 1 - self.crop_left)
        shapes.add_rectangles(
            [[0, left_bdry], [self.im_height, right_bdry]],
            edge_color=TRANSLUCENT_RED,
            face_color=TRANSPARENT,
            edge_width=3,
        )
        self.vis_seg_stack()

    def vis_initiations(self):
        if "initiation" in self.viewer.layers:
            self.viewer.layers.remove("initiation")
        shapes = self.viewer.add_shapes(name="initiation")
        for initiation in self.cur_cell.initiation:
            rel_init = initiation - self.start
            left_bdry = rel_init * (self.crop_right + 1 - self.crop_left)
            right_bdry = (rel_init + 1) * (self.crop_right + 1 - self.crop_left)
            shapes.add_rectangles(
                [[0, left_bdry], [self.im_height, right_bdry]],
                edge_color=TRANSLUCENT_BLUE,
                face_color=TRANSPARENT,
                edge_width=3,
            )

    def mark_initiation(self, viewer: Viewer):
        clicked_cell_id = self.locate_cell_at_cursor()
        if clicked_cell_id is None:
            print(
                "WARNING: tried to add an initiation with a cell that does not exist."
            )
            return
        t, x, y = self.cursor_coords()
        self.cur_cell.initiation.append(t)
        self.cur_cell.initiation_cells.append(clicked_cell_id)
        self.vis_initiations()
        self.vis_seg_stack()

    def remove_initiation(self, viewer: Viewer):
        clicked_cell_id = self.locate_cell_at_cursor()
        if clicked_cell_id is None:
            print("WARNING: tried to remove an initiation that does not exist.")
            return
        t, x, y = self.cursor_coords()
        try:
            self.cur_cell.initiation.remove(t)
            self.cur_cell.initiation_cells.remove(clicked_cell_id)
        except ValueError:
            print("WARNING: tried to remove an initiation that does not exist.")
        self.vis_initiations()
        self.vis_seg_stack()

    def mark_termination(self, viewer: Viewer):
        clicked_cell_id = self.locate_cell_at_cursor()
        if clicked_cell_id is None:
            print(
                "WARNING: tried to add a termination with a cell that does not exist."
            )
            return
        t, x, y = self.cursor_coords()
        self.cur_cell.termination = t
        self.cur_cell.termination_cell = clicked_cell_id
        self.vis_terminal()
        self.vis_seg_stack()

    def locate_cell_at_cursor(self):
        t, x, y = self.cursor_coords()
        seg_stack = load_seg_stack(
            ana_dir=self.analysis_folder,
            experiment_name=self.experiment_name,
            fov_id=self.fov_id,
            peak_id=self.peak_id,
            seg_mode=SegmentationMode.UNET,
        )
        cur_seg_stack = seg_stack[t, :, self.crop_left:self.crop_right + 1]
        # TODO: Proper rounding.
        cur_label = cur_seg_stack[y, x]
        specs = load_specs(self.analysis_folder)
        # need to cache this later!
        mapping = self.all_cells.gen_label_to_cell_mapping(specs)
        if cur_label == 0:
            return None
        try:
            return mapping[self.fov_id][self.peak_id][t][cur_label]
        except KeyError:
            return None

    def set_crop_left(self):
        self.crop_left = self.crop_left_widget.value
        self.update_preview()

    def set_crop_right(self):
        self.crop_right = self.crop_right_widget.value
        self.update_preview()

    def set_cell_generations(self):
        self.num_generations = self.cell_generations_widget.value
        cell_lineage_iter = cell_lineage_filter(
            complete_cell_ids=self.mother_cells.keys(),
            all_cells=self.all_cells,
            generations=self.num_generations,
        )
        self.cell_lineages = list(cell_lineage_iter)
        self.cell_idx = 0
        self.cur_cell_id, self.cur_lineage = self.cell_lineages[self.cell_idx]
        cur_time_range = calc_cell_list_times(self.all_cells, self.cur_lineage)
        self.start = min(cur_time_range)
        self.stop = max(cur_time_range) + 1
        self.fov_id = self.all_cells[self.cur_cell_id].fov
        self.peak_id = self.all_cells[self.cur_cell_id].peak
        self.update_preview()

    def cursor_coords(self):
        coords = self.viewer.cursor.position
        timestamp = (
            round(coords[2] // (self.crop_right + 1 - self.crop_left)) + self.start
        )
        x_coord = round(coords[2]) % (self.crop_right + 1 - self.crop_left)
        y_coord = round(coords[1])
        return timestamp, x_coord, y_coord
 
    def save_to_matlab(self):
        old_cells = read_cells_from_json(self.cell_file_loc)
        write_cells_to_matlab(old_cells, self.cell_file_loc / "all_cells.mat")
