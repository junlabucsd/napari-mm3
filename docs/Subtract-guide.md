# Subtract

This script, according the specs file, creates and average empty channel to be used for subtraction, and then subtracts this empty channel from the specified channel in the phase contrast plane.

**Input**
* TIFF channel stacks (phase contrast only).
* specs.txt file.

**Output**
* Averaged empty stack. Saved in the `empties/` subfolder in the analysis directory.
* Subtracted channel stacks. Saved in the `subtracted/` subfolder in the analysis directory.

## Notes on use

If for a specific FOV there are multiple empty channels designated, then those channels are averaged together by timepoint to create an averaged empty channel. If only one channel is designated in the specs file as empty, then it will simply be copied over. If no channels are designated as empty, than this FOV is skipped, and the user is required to copy one of the empty channels from `empties/` subfolder and rename with the absent FOV ID.
