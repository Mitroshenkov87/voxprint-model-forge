# Computer memory (RAM) tiers

VOXPrint AI Suite sorts computers into three memory tiers by the installed RAM. The tier is shared by all VOXPrint apps.

| Tier | RAM that Windows reports | Typical computer | What the suite does |
| --- | --- | --- | --- |
| **32 GB** | under 44 GB | 32 GB | Models are read from disk each time they are needed and are not kept in memory. |
| **64 GB** | 44 GB to under 88 GB | 48 or 64 GB | Models stay in memory between jobs, and the main ones are prepared for fast transfer to the graphics card. The next model can be loaded in advance. |
| **96 GB** | 88 GB and more | 96 GB or more | All models stay in memory and are prepared for fast transfer to the graphics card. |

A computer reports a little less than its installed memory (a 32 GB computer reports about 31.7 GB), which is why the limits are below the round numbers.

By default the tier follows the installed memory. Advanced users can set it with `memory.ram_mode` (`auto`, `ram32`, `ram64` or `ram96`) in the suite settings file, or with the `VOXPRINT_RAM_MODE` environment variable.

## Model Forge

- Model Forge needs at least **32 GB**. **Build a set** checks for it before it starts, because the models wait in memory between stages.
- In this version Model Forge training works the same on all three tiers. A larger tier helps when other VOXPrint apps run on the same computer.
- In the calibration run on a 32 GB computer, training MOSS-Image Base used at most about 7.6 GB of RAM. If the text encoder has to run on the processor instead of the graphics card (a fallback, see [Compression tests](compression.md)), it needs about 18 GB of RAM.
- **Check this computer** shows the free and total computer memory.
