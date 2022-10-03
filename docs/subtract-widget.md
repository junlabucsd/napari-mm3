## Subtract empty traps from non-empty traps (Subtract widget). 
<img width="1183" alt="subtract" src="https://user-images.githubusercontent.com/40699438/177629512-c5ba4abd-0e03-4540-a4bb-7414ad0560d0.png">

This widget averages empty channel to be used for subtraction, and then subtracts the empty channel from the specified channel in the phase contrast plane.


Downstream analysis of phase contrast (brightfield) images requires background subtraction to remove artifacts of the PDMS device in the images. 

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── specs.yaml
│   └── subtracted
```

**Input**
* TIFF channel stacks (phase contrast only).
* specs.txt file.

**Output**
* Averaged empty stack. Saved in the `empties/` subfolder in the analysis directory.
* Subtracted channel stacks. Saved in the `subtracted/` subfolder in the analysis directory.

### Notes on use

If for a specific FOV there are multiple empty channels designated, then those channels are averaged together by timepoint to create an averaged empty channel. If only one channel is designated in the specs file as empty, then it will simply be copied over. If no channels are designated as empty, than this FOV is skipped, and the user is required to copy one of the empty channels from `empties/` subfolder and rename with the absent FOV ID.

