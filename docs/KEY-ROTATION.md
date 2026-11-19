# Model index signing key rotation

The model index (`index.json` on the Hugging Face dataset `Mitroshenkov87/voxprint-model-index`) is signed on the Model Forge box. `voxprint-forge-cli index push --push` reads the Ed25519 private key from `VOXPRINT_MDL_INDEX_KEY_FILE` (or `--key-file`), signs the exact bytes of `index.json`, and uploads `index.json` and `index.json.sig` in one commit. The signature file is base64 of the 64-byte signature, the same format as `voxprint-mdl index sign`. The private key is never committed or printed. The public key is in `keys/index-public.pem` and is bundled in voxprint-common as `index-public.pem`.

## When the key is lost or leaked

1. Generate a new key on the box:
   `openssl genpkey -algorithm ED25519 -out /workspace/.vm-access/mdl-index-key.pem`, then `chmod 600` it and write the public half with `openssl pkey -in ... -pubout -out /workspace/.vm-access/mdl-index-public.pem`.
2. Replace `keys/index-public.pem` here and send the new public key to Components so voxprint-common ships it in `index-public.pem`. For a planned rotation, ship both keys (one PEM block after the other) and drop the old one in a later release. For a leaked key, ship only the new key.
3. Rebuild the index with `index build`; the serial goes up by one over the published serial.
4. Re-sign and publish with `index push --push`, then check it with `voxprint-mdl index verify index.json --public-key keys/index-public.pem`.

Apps keep using the index bundled in their voxprint-common release until they update to a release with the new key. A signature they cannot verify makes them fall back to that bundled index, and the download still runs. A leaked old key stays trusted by apps that have not updated, so ship the voxprint-common release quickly.
