# Training

## Step 4: Check and prepare

![Step 4, Check and prepare](../img/en-simple-4-prepare.png)

1. Choose the crop: **Focus on subject** or **Keep whole photo**.
2. Press **Check photos**. The app checks each photo, crops it and writes a short description of it. The trigger word is added to every description, so the model links the word to your subject.
3. Read the list. **Ready for training** shows the photos that will be used with their descriptions. **Set aside** lists the photos that will not be used, each with a reason (for example, too blurry or almost the same as another photo).

You can remove any photo before training. If too few photos are left, add more on step 1.

## Step 5: Train

Choose a preset, a likeness and the graphics card memory (see [Presets](presets.md)), and the base model:

- **Base model** is the picture model the LoRA is trained on. The LoRA works only with this base, so pick the same one in Canvas. **Z-Image** is best for maximum quality and is slower. **MOSS-Image Base** is faster and fits a 16 GB card. The second line of each option shows what it is best for and its video memory.

  ![The base model list open](../img/en-simple-5-train-base-open.png)

- **Details ▸** under each setting explains it. The base model details include the measured video memory.

  ![Step 5 with Details open](../img/en-simple-5-train-details-open.png)

- **Show test pictures** is a switch. While it is on, the app draws a few pictures during training so you can see progress. Those pictures are separate from the ones you mark between rounds.

Press **Estimate time** for a forecast, then **Start training**. While a stage is running, the step says that training is running, and the preset, likeness, memory and base model stay locked. A line shows the step out of the total, how long the stage has been running and the time left, in hours and minutes. The bar under that line ends with the percent. The loss is under **Details**. **Pause** and **Stop** stay available. **Stop** asks before it ends the run.

![Training in progress](../img/en-simple-5-train-running.png)

- **Pause** saves a version and stops. Press **Continue** later to go on from that version.
- **Stop** asks, then ends the run. The versions saved so far are kept.
- The app saves a version six times during a run, so you can compare them on the next step.

## Review rounds

With **Extend the set from samples** on (Photos step), training pauses after each stage. The app saves a checkpoint and draws a fixed set of pictures from it. The heading is **Check the new pictures**. Your photos stay beside the new pictures. Each new picture has **Include** and **Leave out**. A picture you include gets a gold border and a check. A picture you leave out is dimmed, with a red border and a cross. A picture with no mark stays plain.

![A review round](../img/en-simple-5-review.png)

**Mark all 👍** includes every new picture. **Reset** clears the marks. The footer counts pictures you included, pictures you left out, and pictures with no mark. **Continue training** starts the next pass on your photos and the pictures you liked. If any picture has no mark, the app asks you to confirm: unmarked pictures will not join the set. **Finish training** saves the model now and ends the rounds. You can finish before the last planned round. Back and Next stay hidden; Continue and Finish drive this screen. Closing the app does not drop a paused review: open the project again and the same screen is there.

On the command line, `lora train --review-rounds N --samples-per-round K` pauses the same way. `lora review` lists the pictures and sets marks (`--like`, `--dislike`, or `--from-file`). `lora continue` starts the next stage and `lora finish` ends the rounds. A train without those options runs straight through, as before.

## Expert settings

In Expert mode the panel on the right shows every value: Rank, Alpha, Steps, Learning rate, Scheduler, Warm-up steps, Batch size, Gradient accumulation, Resolution, Precision, Gradient checkpointing, Train the text encoder, Descriptions, Description model, Caption dropout, Seed, Save every N steps and Test picture every N steps. The reset button next to the presets puts back the values of the chosen preset.

![Expert mode while training](../img/en-expert-5-train-running.png)

Recommended: 10 to 30 photos. Allowed in Expert: 5 to 60 photos.
