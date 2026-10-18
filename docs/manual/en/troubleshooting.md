# Troubleshooting, privacy and licences

## Troubleshooting

| What you see | What to do |
| --- | --- |
| "Training needs a supported graphics card" | Use a computer with an NVIDIA card with 16 GB or more and a current driver. |
| "Models: not installed yet" | Press **Manage models** and download the missing models. |
| Preparation does not start because models are missing | Download Qwen3-VL-4B-Instruct, MiVOLO v2 and YuNet in Manage models. |
| "Add at least N more photos" | Many photos were set aside. Add more clear photos, or use Expert mode. |
| "These photos cannot be used for training" | Some photos were refused and this project cannot continue. Start a new project with other photos of consenting adults. |
| "A PAUSE file is still in the project folder" | Press **Continue** in the app. If it repeats, delete the file named `PAUSE` in the project folder. |
| Out of video memory | Choose **16 GB** under Graphics card memory, close other programs that use the graphics card, and try again. |
| Not enough disk space | Free space on the drive that holds `%LOCALAPPDATA%\VOXPrint AI Suite`, or remove models you no longer need in Manage models. |
| The setup failed | Read `C:\Program Files\VOXPrint AI Suite\Model Forge\setup-work\install.log`, check the internet connection and run the setup again. |
| Test pictures do not look like the subject | Use **Close** likeness or **Best quality**, add more varied photos, or choose a later version. |
| Every picture looks the same as the photos | Use **Flexible** likeness or an earlier version, and add more variety to the photos. |

## FAQ

**Can I train on photos of someone I found online?** No. Train only on consenting adults and on photos you have the right to use.

**Why were some photos set aside or refused?** Step 4 shows a reason for each photo that is set aside. Some photos may be refused; replace them with other photos.

**Does anything leave my computer?** No. Only the app libraries and the models are downloaded from the internet.

**Can I use the LoRA commercially?** The LoRA follows the licence of its base model. Z-Image and MOSS-Image Base are Apache-2.0. You are also responsible for the rights to the photos and the consent of the people in them.

**Can I continue training later?** Yes. Pause and continue, or use **Add more photos** on a finished model.

## Privacy

- Training, descriptions and checks run on your computer. Photos, descriptions and models are never uploaded.
- The internet is used only to download the app libraries and models from their official sources.
- Uninstalling the app keeps your data. To remove everything, also delete `%LOCALAPPDATA%\VOXPrint AI Suite\state\model-forge` and, if no other VOXPrint app uses them, the models in `%LOCALAPPDATA%\VOXPrint AI Suite\models`.

## Licences

| Component | Licence |
| --- | --- |
| VOXPrint AI Model Forge | Apache-2.0 |
| Z-Image, Z-Image-Turbo | Apache-2.0 |
| MOSS-Image Base | Apache-2.0 |
| Qwen3-VL-4B-Instruct | Apache-2.0 |
| Florence-2-large | MIT |
| MiVOLO v2 | Apache-2.0 |
| YuNet | MIT |
| DINOv2 base | Apache-2.0 |

The full licence texts come with each model and are listed in Manage models. The app's own notice is in `LICENSE` and `NOTICE` in the install folder.
