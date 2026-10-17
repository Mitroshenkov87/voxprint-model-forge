"""VOXPrint AI Model Forge user app on the shared Yitro shell (voxprint-common v0.2.5).

``draft`` and ``engine`` are Qt-free. ``window`` builds the screen. The LoRA
work itself runs in ``voxprint-forge-cli lora ... --json``; this package only
collects the inputs, shows the engine's plain-words progress and never
decides anything about safety.
"""
